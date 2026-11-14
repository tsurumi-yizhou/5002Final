"""Offline search distillation into the single-file competition program.

Example (from the project root):
  python training/train.py --output-dir training/run --cycles 1 --export
  python training/train.py --output-dir training/run --forever --export
  python training/train.py --output-dir training/run --evaluate-only

Finite training ends with an independent tournament; --forever evaluates every
--evaluate-every cycles (default 10). Each tournament defaults to 50 fresh boards,
played in both colours against the prelearning reference. Repeat --eval-opponent
PATH to select other opponents. --eval-boards controls the number of boards.
--evaluate-only evaluates the latest fitted checkpoint without training it again.
Evaluation records freeze both players' complete source and the fitted weights.
An interrupted tournament resumes its saved plan before further training; running
evaluation again after completion allocates new seeds. Evaluation seeds occupy
residues excluded from all training/validation/test families, across every cycle.
Results live in evaluation/batch-*/results.json and never enter data.jsonl or fit().

Each cycle adds new seeds to disjoint training/validation/test families. Random
legal openings lead into fast full self-play games. Spaced, difficult search
positions are relabelled with longer MCTS; its visit distribution is an empirical
teacher, while explicit tactical proofs are tagged separately. A lost game is
never used as a label for every preceding move. Proven lost positions are skipped.

Only TRAIN updates coefficients; VALIDATION chooses the epoch and prior strength.
TEST is reported afterwards. Features encode relative colours, connected-window
boundaries and reflection symmetry. Quantized, supported coefficients are embedded
as Python literals by --export. Zero validation-selected strength keeps them dormant.
The submission uses only its existing NumPy/Pygame/time imports and no data files.

The learning rate is --learning-rate / sqrt(1 + epoch / --lr-decay-epochs),
defaulting to 0.12 / sqrt(1 + epoch / 50). Each fit reports its last learning rate
and the rate at its selected epoch. Validation is checked every ten epochs:
--fit-patience checks without --fit-min-delta improvement end the fit, retaining
the best quantized weights. Defaults are 3 checks and 0.0001 cross-entropy units.
Across cycles, the previous and new models are evaluated on the SAME current
validation examples. --cycle-patience successive gains no larger than
--cycle-min-delta stop the training loop (defaults 5 cycles and 0.001 units).
Either patience may be set to zero to disable that stopping rule. Test positions
and independent matches never decide early stopping. A loop stop also runs its
final independent tournament. Stop counters survive interruption; explicitly
starting training again after a completed early stop opens a new patience window.

Completed games and partially labelled games resume from progress.json; data.jsonl
holds teacher examples; metrics.json holds the latest fit. Each invocation adds
--cycles cycles from its last completed cycle, unless early stopping intervenes.
--forever continues until Ctrl-C or cycle-level early stopping;
completed examples survive, and an interrupted game restarts. Default inference
budgets are unchanged: accelerated clocks are used only by this offline trainer.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from run import MOVE_LIMIT, exact_five, load_agent, normalize_point, play_game, save_results, summarize

FEATURES = 81920


def now():
    return datetime.now(timezone.utc).isoformat()


def agent(source, name, scale):
    module = load_agent(source, name)
    real_counter = time.perf_counter
    module.time = SimpleNamespace(perf_counter=lambda: real_counter() / scale)
    module.best_move._policy_data_override = ()
    module.best_move._policy_scale_override = 0.0
    return module


def call(module, state, player, seed):
    records = []
    module.best_move._training_recorder = records.append
    np.random.seed(seed)
    supplied = tuple(array.copy() for array in state)
    started = time.perf_counter()
    value = module.best_move(supplied, player)
    seconds = time.perf_counter() - started
    point = normalize_point(value)
    if (point is None or not (0 <= point[0] < state[0].shape[0] and
                             0 <= point[1] < state[0].shape[1]) or
            state[0][tuple(point)] != 0):
        raise ValueError("Agent returned an illegal move")
    if any(not np.array_equal(a, b) for a, b in zip(supplied, state)):
        raise ValueError("Agent changed its supplied input")
    if not records:
        raise ValueError("Source does not expose the training recorder")
    return point, seconds, records[-1]


def random_seed(*values):
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def collect_game(module, split, seed, args):
    rng = np.random.default_rng(seed)
    state = (np.zeros((17, 13), dtype=np.int_),
             rng.random((17, 12)) < .85, rng.random((16, 13)) < .7)
    history, snapshots = [], []
    seconds = 0.0
    maximum = 0.0
    for ply in range(1, 222):
        player = 1 if ply % 2 else -1
        if ply <= args.random_opening:
            empty = np.flatnonzero(state[0].reshape(-1) == 0)
            point = list(divmod(int(rng.choice(empty)), 13))
            elapsed, record = 0.0, None
        else:
            point, elapsed, record = call(module, state, player,
                                          random_seed(seed, ply, 0))
        seconds += elapsed
        maximum = max(maximum, elapsed)
        if record and record["reason"] == "search" and record["policy_features"]:
            visits = sorted((child["visits"] for child in record["children"]), reverse=True)
            margin = ((visits[0] - visits[1]) / max(1, visits[0])
                      if len(visits) > 1 else 1.0)
            snapshots.append({
                "id": f"{split}:{seed}:{ply}", "split": split, "seed": seed,
                "ply": ply, "player": player,
                "state": [a.astype(int).tolist() for a in state],
                "features": record["policy_features"], "base": record["base_logits"],
                "student_move": point[0] * 13 + point[1], "student_seconds": elapsed,
                "student_iterations": record["iterations"], "visit_margin": margin,
                "difficulty": elapsed / max(.001, .5 * args.fast_scale) + 1 - margin,
            })
        state[0][tuple(point)] = player
        history.append(point)
        if exact_five(state, point, player):
            winner, reason = player, "exact_five"
            break
        if ply == 221:
            winner, reason = -1, "full_board_white"
            break
    selected = snapshots[:1]  # retain early context as well as later hard positions
    for item in sorted(snapshots[1:], key=lambda row: (-row["difficulty"], row["ply"])):
        if len(selected) >= args.positions_per_game:
            break
        if all(abs(item["ply"] - previous["ply"]) >= 5 for previous in selected):
            selected.append(item)
    for item in snapshots:
        if len(selected) >= args.positions_per_game:
            break
        if all(item["id"] != previous["id"] for previous in selected):
            selected.append(item)
    summary = {"split": split, "seed": seed, "plies": ply, "winner": winner,
               "reason": reason, "seconds": seconds, "maximum_call_seconds": maximum,
               "search_positions": len(snapshots), "selected_positions": len(selected),
               "moves": history}
    print(f"Collected {split} seed={seed}: {ply} plies, winner={winner:+d}, "
          f"{len(selected)} positions, {seconds:.2f}s", flush=True)
    return {"summary": summary, "positions": selected}


def teach(module, snapshot, args):
    state = tuple(np.asarray(array, dtype=np.int_ if index == 0 else np.bool_)
                  for index, array in enumerate(snapshot["state"]))
    player = snapshot["player"]
    features = {int(key): value for key, value in snapshot["features"].items()}
    accumulated, teacher_moves, elapsed = {}, [], 0.0
    provenance = "search_distribution"
    result = dict(snapshot)
    for repeat in range(args.teacher_repeats):
        point, seconds, record = call(module, state, player,
                                      random_seed(snapshot["seed"], snapshot["ply"], repeat, 1))
        elapsed += seconds
        move = point[0] * state[0].shape[1] + point[1]
        teacher_moves.append(move)
        if record["solved"] == -player:
            result.update(target=None, skipped="teacher_proved_loss", teacher_seconds=elapsed)
            return result
        if move not in features:
            result.update(target=None, skipped="teacher_choice_outside_student_candidates",
                          teacher_seconds=elapsed)
            return result
        if record["solved"] == player:
            accumulated = {move: 1.0}
            provenance = "proved_winning_choice"
            break
        distribution = {child["move"]: child["visits"] for child in record["children"]
                        if child["move"] in features and child["visits"] > 0
                        and child["solved"] != -player}
        total = sum(distribution.values())
        if not total:
            distribution, total = {move: 1}, 1
        for key, value in distribution.items():
            accumulated[key] = accumulated.get(key, 0.0) + value / total
    total = sum(accumulated.values())
    result.update(target={key: value / total for key, value in accumulated.items()},
                  teacher_moves=teacher_moves, teacher_seconds=elapsed,
                  provenance=provenance, skipped=None)
    return result


def prepare(row):
    features = {int(key): value for key, value in row["features"].items()}
    base = {int(key): value for key, value in row["base"].items()}
    targets = {int(key): value for key, value in row["target"].items()}
    moves = sorted(features)
    width = max(1, max(map(len, features.values())))
    indices = np.zeros((len(moves), width), dtype=np.int_)
    factors = np.zeros_like(indices, dtype=float)
    for index, move in enumerate(moves):
        keys = features[move]
        indices[index, :len(keys)] = keys
        factors[index, :len(keys)] = 1.0 / max(1, len(keys)) ** .5
    return {"moves": moves, "indices": indices, "factors": factors,
            "base": np.array([base[move] for move in moves]),
            "target": np.array([targets.get(move, 0.0) for move in moves])}


def distribution(example, weights, scale):
    raw = (weights[example["indices"]] * example["factors"]).sum(axis=1)
    logits = example["base"] + scale * np.clip(raw, -.8, .8)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs, raw


def evaluate(examples, weights, scale):
    if not examples:
        return {"positions": 0, "cross_entropy": None, "top1_agreement": None,
                "top3_agreement": None}
    losses, top1, top3 = [], 0, 0
    for example in examples:
        probs, _ = distribution(example, weights, scale)
        target = example["target"]
        losses.append(float(-(target * np.log(np.maximum(probs, 1e-15))).sum()))
        teacher = int(np.argmax(target))
        order = np.argsort(-probs, kind="stable")
        top1 += int(order[0] == teacher)
        top3 += int(teacher in order[:3])
    return {"positions": len(examples), "cross_entropy": float(np.mean(losses)),
            "top1_agreement": top1 / len(examples), "top3_agreement": top3 / len(examples)}


def quantize(weights, support, args):
    eligible = np.flatnonzero((support >= args.min_support) & (np.abs(weights) >= .5 / 256))
    if len(eligible) > args.max_entries:
        ranking = np.argsort(-np.abs(weights[eligible]) * np.sqrt(support[eligible]), kind="stable")
        eligible = eligible[ranking[:args.max_entries]]
    dense = np.zeros(FEATURES, dtype=float)
    quantized = np.clip(np.rint(weights[eligible] * 256), -32768, 32767).astype(int)
    pairs = sorted((int(key), int(value)) for key, value in zip(eligible, quantized) if value)
    for key, value in pairs:
        dense[key] = value / 256
    return tuple(pairs), dense


def learning_rate_at(args, epoch):
    return args.learning_rate / (1 + epoch / args.lr_decay_epochs) ** .5


def fit(rows, args, previous=None):
    splits = {name: [prepare(row) for row in rows if row["split"] == name and row.get("target")]
              for name in ("train", "validation", "test")}
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("Need labelled positions in all three seed-disjoint splits")
    weights = np.zeros(FEATURES, dtype=float)
    support = np.zeros(FEATURES, dtype=np.int_)
    for example in splits["train"]:
        support[np.unique(example["indices"][example["factors"] != 0])] += 1
    eligible = support >= args.min_support
    before = {name: evaluate(examples, weights, 0) for name, examples in splits.items()}
    best_loss = before["validation"]["cross_entropy"]
    best, best_epoch, best_scale = (), 0, 0.0
    latest = ()
    significant_loss, stale_checks = best_loss, 0
    history = []
    early_stopped = False
    rng = np.random.default_rng(args.fit_seed)
    for epoch in range(1, args.epochs + 1):
        learning_rate = learning_rate_at(args, epoch)
        for index in rng.permutation(len(splits["train"])):
            example = splits["train"][int(index)]
            probs, raw = distribution(example, weights, 1.0)
            gradient = (example["target"] - probs) * (np.abs(raw) < .8)
            delta = learning_rate * gradient[:, None] * example["factors"]
            delta *= eligible[example["indices"]]
            np.add.at(weights, example["indices"].ravel(), delta.ravel())
            weights[eligible] *= .999
        if epoch % 10 == 0 or epoch == args.epochs:
            latest, dense = quantize(weights, support, args)
            for scale in (.25, .5, 1.0):
                loss = evaluate(splits["validation"], dense, scale)["cross_entropy"]
                if loss < best_loss - 1e-10:
                    best_loss, best, best_epoch, best_scale = loss, latest, epoch, scale
            if significant_loss - best_loss > args.fit_min_delta:
                significant_loss, stale_checks = best_loss, 0
            else:
                stale_checks += 1
            early_stopped = bool(args.fit_patience and stale_checks >= args.fit_patience
                                 and epoch < args.epochs)
            history.append({"epoch": epoch, "learning_rate": learning_rate,
                            "best_validation_cross_entropy": best_loss,
                            "stale_checks": stale_checks})
            print(f"Fit epoch {epoch}/{args.epochs}: learning_rate={learning_rate:.6f}, "
                  f"best validation CE={best_loss:.6f}, "
                  f"stale checks={stale_checks}/{args.fit_patience or 'disabled'}", flush=True)
            if early_stopped:
                break
    # Keep an inspectable trained table even when validation selects no influence.
    if best_scale == 0:
        best = latest
    selected = np.zeros(FEATURES, dtype=float)
    for key, value in best:
        selected[key] = value / 256
    after = {name: evaluate(examples, selected, best_scale) for name, examples in splits.items()}
    comparison = None
    if previous is not None:
        previous_weights = np.zeros(FEATURES, dtype=float)
        for key, value in previous["coefficients"]:
            previous_weights[key] = value / 256
        previous_loss = evaluate(splits["validation"], previous_weights,
                                 previous["selected_scale"])["cross_entropy"]
        comparison = {"previous_cycle": previous["cycle"],
                      "validation_positions": len(splits["validation"]),
                      "previous_cross_entropy": previous_loss,
                      "current_cross_entropy": after["validation"]["cross_entropy"],
                      "improvement": previous_loss - after["validation"]["cross_entropy"]}
    return best, best_scale, {
        "before": before, "after": after, "selected_epoch": best_epoch,
        "selected_scale": best_scale, "entries": len(best),
        "dense_lookup_bytes": FEATURES * 2, "support_from": "distinct TRAIN positions only",
        "validation_selection": "minimum quantized-model cross entropy; test excluded",
        "target_meaning": "longer search visit distributions or tagged tactical winning choices",
        "optimization": {
            "initial_learning_rate": args.learning_rate,
            "decay_epochs": args.lr_decay_epochs,
            "last_learning_rate": learning_rate,
            "selected_learning_rate": learning_rate_at(args, best_epoch) if best_epoch else None,
            "epochs_completed": epoch, "epoch_limit": args.epochs,
            "early_stopped": early_stopped,
            "stop_reason": "validation_plateau" if early_stopped else "epoch_limit",
            "patience_checks": args.fit_patience, "min_delta": args.fit_min_delta,
            "history": history},
        "cycle_comparison": comparison,
    }


def decide_training_stop(metrics, progress, args):
    comparison = metrics["cycle_comparison"]
    stale = progress.get("training_decision", {}).get("stale_cycles", 0)
    if comparison is None or not args.cycle_patience:
        stale = 0
    elif comparison["improvement"] > args.cycle_min_delta:
        stale = 0
    else:
        stale += 1
    stop = bool(args.cycle_patience and stale >= args.cycle_patience)
    reason = ("disabled" if not args.cycle_patience else
              "first_checkpoint" if comparison is None else
              "validation_plateau" if stop else
              "validation_improved" if stale == 0 else "waiting_for_patience")
    return {"stop": stop, "reason": reason, "stale_cycles": stale,
            "patience_cycles": args.cycle_patience, "min_delta": args.cycle_min_delta,
            "validation_comparison": comparison,
            "uses_test_or_match_results": False}


def print_training_status(checkpoint):
    optimization = checkpoint.get("optimization")
    decision = checkpoint.get("training_decision")
    if optimization is None:
        print("Learning rate / early stopping: not recorded for this checkpoint.", flush=True)
        return
    selected = optimization["selected_learning_rate"]
    selected_text = f"{selected:.6f}" if selected is not None else "inactive policy"
    print(f"Learning rate: selected={selected_text}, "
          f"last={optimization['last_learning_rate']:.6f}; "
          f"epochs={optimization['epochs_completed']}/{optimization['epoch_limit']}", flush=True)
    print(f"Fit decision: {'EARLY STOP' if optimization['early_stopped'] else 'EPOCH LIMIT'} "
          f"({optimization['stop_reason']}); selected epoch={checkpoint['selected_epoch']}", flush=True)
    if decision is not None:
        comparison = decision["validation_comparison"]
        gain = (f"validation gain={comparison['improvement']:.6f} on "
                f"{comparison['validation_positions']} shared positions; " if comparison else "")
        print(f"Training decision: {'EARLY STOP' if decision['stop'] else 'CONTINUE'}; "
              f"{gain}stale cycles={decision['stale_cycles']}/"
              f"{decision['patience_cycles'] or 'disabled'} ({decision['reason']})", flush=True)


def embedded_source(text, pairs, scale):
    # JSON stores pairs as lists; tuple literals preserve the inference cache
    # identity across moves, exactly as in the exported competition program.
    pairs = tuple((int(key), int(weight)) for key, weight in pairs)
    tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "best_move")
    assignments = {node.targets[0].id: node for node in function.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in ("_learned_policy_data", "_learned_policy_scale")}
    if len(assignments) != 2:
        raise ValueError("Cannot locate the two embedded policy literals")
    lines = text.splitlines(keepends=True)
    data_lines = ["    _learned_policy_data = (\n"]
    for offset in range(0, len(pairs), 6):
        data_lines.append("        " + ", ".join(repr(pair) for pair in pairs[offset:offset + 6]) + ",\n")
    data_lines.append("    )\n")
    replacements = {
        "_learned_policy_data": "".join(data_lines) if pairs else "    _learned_policy_data = ()\n",
        "_learned_policy_scale": f"    _learned_policy_scale = {scale!r}\n",
    }
    for name, node in sorted(assignments.items(), key=lambda item: item[1].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[name]]
    updated = "".join(lines)
    ast.parse(updated)
    return updated


def export(source, pairs, scale):
    updated = embedded_source(source.read_text(), pairs, scale)
    temporary = source.with_name(source.name + ".training-tmp")
    temporary.write_text(updated)
    os.replace(temporary, source)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def evaluation_records(output_dir):
    return sorted((output_dir / "evaluation").glob("batch-*/results.json"))


def pending_evaluation(output_dir):
    records = evaluation_records(output_dir)
    if records:
        latest = json.loads(records[-1].read_text())
        if latest["status"] != "complete":
            return records[-1], latest
    return None


def frozen_agent(snapshot, name):
    """Load the saved source, including its weights, using the real match clock."""
    module = ModuleType(name)
    module.__file__ = snapshot["source_path"]
    sys.modules[name] = module
    exec(compile(snapshot["source_text"], module.__file__, "exec"), module.__dict__)
    return module


def evaluation_summary(result):
    games, seeds = result["games"], result["settings"]["seeds"]
    opponents = {}
    for opponent in result["frozen_agents"]["opponents"]:
        played = [game for game in games if game["opponent_id"] == opponent["id"]]
        stats = summarize(played, seeds)
        stats["source_path"] = opponent["source_path"]
        stats["candidate_win_rate"] = (stats["candidate_raw_round_wins"] / len(played)
                                       if played else None)
        opponents[opponent["id"]] = stats
    wins = sum(game["winner"] == "candidate" for game in games)
    return {"completed_games": len(games),
            "planned_games": 2 * len(seeds) * len(opponents),
            "candidate_wins": wins,
            "candidate_win_rate": wins / len(games) if games else None,
            "opponents": opponents}


def evaluate_checkpoint(args, configuration):
    pending = pending_evaluation(args.output_dir)
    if pending:
        path, result = pending
        print(f"Resuming frozen evaluation: {path}", flush=True)
    else:
        metrics_path = args.output_dir / "metrics.json"
        if not metrics_path.exists():
            raise ValueError("Train a checkpoint before running independent evaluation")
        metrics = json.loads(metrics_path.read_text())
        excluded = {seed for family in configuration["families"].values() for seed in family}
        available = [seed for seed in range(100000) if seed not in excluded]
        if args.eval_boards > len(available):
            raise ValueError("Too many evaluation boards for the reserved seed space")
        previous = evaluation_records(args.output_dir)
        batch = int(previous[-1].parent.name.split("-")[1]) + 1 if previous else 0
        seeds = [batch * 100000 + seed for seed in available[-args.eval_boards:]]
        path = args.output_dir / "evaluation" / f"batch-{batch:08d}" / "results.json"
        opponents = args.eval_opponent or [ROOT / "benchmarks/prelearning_reference.py"]
        frozen = {"candidate": {
            "source_path": str(args.source),
            "source_text": embedded_source(args.source.read_text(),
                                           metrics["coefficients"], metrics["selected_scale"])},
            "opponents": [{"id": f"opponent_{index + 1}",
                           "source_path": str(source.resolve()),
                           "source_text": source.read_text()}
                          for index, source in enumerate(opponents)]}
        result = {
            "measurement_type": "independent_frozen_policy_tournament",
            "status": "running", "started_at": now(),
            "training_checkpoint": {key: metrics[key] for key in
                                    ("cycle", "updated_at", "selected_epoch",
                                     "selected_scale", "entries")},
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "platform": platform.platform()},
            "settings": {"seeds": seeds, "rows": 17, "cols": 13,
                         "horizontal": .85, "vertical": .7,
                         "move_limit_seconds": MOVE_LIMIT, "parallel_games": False,
                         "training_seed_residues": sorted(excluded),
                         "seed_rule": "batch * 100000 + reserved non-training residue",
                         "board_rng": "numpy.Generator(PCG64(seed))",
                         "agent_rng": "SeedSequence([seed, ply, color == 1, agent_index])",
                         "agent_indices": {"candidate": 0, "baseline": 1},
                         "match_tiebreak": "lower combined thinking time across both rounds"},
            "frozen_agents": frozen, "games": []}
        result["training_checkpoint"]["optimization"] = metrics.get("optimization")
        result["training_checkpoint"]["training_decision"] = metrics.get("training_decision")
        result["summary"] = evaluation_summary(result)
        save_results(path, result)
        print(f"Frozen cycle {metrics['cycle']}: {len(seeds)} new boards, "
              f"{len(opponents)} opponent(s), {result['summary']['planned_games']} games. "
              f"Saved {path}", flush=True)

    try:
        result["status"] = "running"
        candidate = frozen_agent(result["frozen_agents"]["candidate"], "evaluation_candidate")
        opponents = {item["id"]: frozen_agent(item, "evaluation_" + item["id"])
                     for item in result["frozen_agents"]["opponents"]}
        completed = {(game["opponent_id"], game["seed"], game["candidate_color"])
                     for game in result["games"]}
        for opponent_id, opponent in opponents.items():
            modules = {"candidate": candidate, "baseline": opponent}
            for seed in result["settings"]["seeds"]:
                for color in (1, -1):
                    if (opponent_id, seed, color) in completed:
                        continue
                    game = play_game(modules, result["settings"], seed, color,
                                     len(result["games"]) + 1)
                    game["opponent_id"] = opponent_id
                    result["games"].append(game)
                    result["summary"] = evaluation_summary(result)
                    result["updated_at"] = now()
                    save_results(path, result)
                    print(f"Evaluation {len(result['games'])}/"
                          f"{result['summary']['planned_games']}: {opponent_id}, "
                          f"{game['winner']} wins ({game['reason']}). Saved {path}", flush=True)
        result["status"] = "complete"
        result["updated_at"] = now()
        save_results(path, result)
        for opponent_id, stats in result["summary"]["opponents"].items():
            black, white = (stats["candidate_by_color"][color] for color in ("black", "white"))
            timing = stats["agents"]["candidate"]
            print(f"{opponent_id}: wins {stats['candidate_raw_round_wins']}/"
                  f"{stats['completed_games']} ({stats['candidate_win_rate']:.1%}); "
                  f"Black {black['wins']}/{black['games']}, White {white['wins']}/{white['games']}; "
                  f"points {stats['official_points']}; mean move "
                  f"{timing['mean_seconds_per_move']:.3f}s, max {timing['move_max_seconds']:.3f}s",
                  flush=True)
        print(f"Independent evaluation complete: {path}", flush=True)
        print_training_status(result["training_checkpoint"])
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["updated_at"] = now()
        save_results(path, result)
        print(f"Evaluation stopped; completed games and frozen players saved in {path}", flush=True)
        raise
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "gomoku.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/run")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[31001, 31002, 31003, 31004, 31005, 31006])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[41001, 41002])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[51001, 51002])
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="evaluate the latest fit, or resume its interrupted tournament")
    parser.add_argument("--eval-boards", type=int, default=50,
                        help="fresh boards per evaluation, each played in both colours (default 50)")
    parser.add_argument("--eval-opponent", type=Path, action="append",
                        help="opponent source; repeat to evaluate against multiple strategies")
    parser.add_argument("--evaluate-every", type=int, default=10,
                        help="with --forever, evaluate every N completed training cycles")
    parser.add_argument("--fast-scale", type=float, default=.15)
    parser.add_argument("--teacher-scale", type=float, default=1.5)
    parser.add_argument("--teacher-repeats", type=int, default=1)
    parser.add_argument("--positions-per-game", type=int, default=3)
    parser.add_argument("--random-opening", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=.12)
    parser.add_argument("--lr-decay-epochs", type=float, default=50)
    parser.add_argument("--fit-patience", type=int, default=3,
                        help="validation checks without progress before ending a fit; 0 disables")
    parser.add_argument("--fit-min-delta", type=float, default=.0001)
    parser.add_argument("--cycle-patience", type=int, default=5,
                        help="cycles without validation progress before stopping training; 0 disables")
    parser.add_argument("--cycle-min-delta", type=float, default=.001)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-entries", type=int, default=2048)
    parser.add_argument("--fit-seed", type=int, default=7231)
    args = parser.parse_args()
    if args.eval_boards < 2 or args.eval_boards % 2 or args.evaluate_every < 1:
        parser.error("eval-boards must be even and at least 2; evaluate-every must be positive")
    if args.evaluate_only and args.forever:
        parser.error("evaluate-only and forever select different workflows")
    rates = (args.learning_rate, args.lr_decay_epochs, args.fit_min_delta, args.cycle_min_delta)
    if (not all(np.isfinite(value) for value in rates) or
            min(args.learning_rate, args.lr_decay_epochs) <= 0 or
            min(args.fit_patience, args.cycle_patience, args.fit_min_delta, args.cycle_min_delta) < 0):
        parser.error("learning rate and decay must be positive; patience and min-delta nonnegative")
    families = {"train": args.train_seeds, "validation": args.validation_seeds, "test": args.test_seeds}
    seeds = [seed for values in families.values() for seed in values]
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 100000 for seed in seeds):
        parser.error("seed families must be disjoint, unique and between 0 and 99999")
    if (min(args.cycles, args.teacher_repeats, args.positions_per_game, args.epochs,
            args.min_support, args.max_entries) < 1 or
            not 0 < args.fast_scale < args.teacher_scale or not 0 <= args.random_opening < 221):
        parser.error("use positive counts, teacher-scale > fast-scale > 0, and opening < 221")
    args.source, args.output_dir = args.source.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path, data_path = args.output_dir / "progress.json", args.output_dir / "data.jsonl"
    configuration = {"source": str(args.source), "families": families,
                     "fast_scale": args.fast_scale, "teacher_scale": args.teacher_scale,
                     "teacher_repeats": args.teacher_repeats,
                     "positions_per_game": args.positions_per_game,
                     "random_opening": args.random_opening}
    progress = (json.loads(progress_path.read_text()) if progress_path.exists() else
                {"configuration": configuration, "next_cycle": 0, "completed_games": {}, "pending": None})
    if progress["configuration"] != configuration:
        parser.error("use the output directory's saved collection settings or a new directory")
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    if args.evaluate_only or pending_evaluation(args.output_dir):
        try:
            evaluation_path = evaluate_checkpoint(args, configuration)
        except KeyboardInterrupt:
            return 130
        if args.evaluate_only:
            return 0
        finished = json.loads(evaluation_path.read_text())
        if (finished["training_checkpoint"].get("training_decision") or {}).get("stop"):
            return 0
    if progress.get("training_decision", {}).get("stop"):
        progress.pop("training_decision")
        print("Explicit training restart: starting a new early-stop patience window.", flush=True)
    student = agent(args.source, "training_student", args.fast_scale)
    teacher = agent(args.source, "training_teacher", args.teacher_scale)
    rows = read_rows(data_path)
    known = {row["id"] for row in rows}
    previous_metrics = args.output_dir / "metrics.json"
    previous = None
    if progress["next_cycle"] and previous_metrics.exists():
        saved = json.loads(previous_metrics.read_text())
        previous = saved
        student.best_move._policy_data_override = tuple(tuple(pair) for pair in saved["coefficients"])
        student.best_move._policy_scale_override = saved["selected_scale"]
    end_cycle = progress["next_cycle"] + args.cycles
    try:
        while args.forever or progress["next_cycle"] < end_cycle:
            cycle = progress["next_cycle"]
            print(f"Cycle {cycle}: collecting new seed families", flush=True)
            for split, family in families.items():
                for initial in family:
                    seed = initial + cycle * 100000
                    game_id = f"{split}:{seed}"
                    if game_id in progress["completed_games"]:
                        continue
                    if progress["pending"] is None:
                        progress["pending"] = collect_game(student, split, seed, args)
                        save_results(progress_path, progress)
                    game = progress["pending"]
                    if game["summary"]["seed"] != seed or game["summary"]["split"] != split:
                        raise ValueError("Pending game differs from the next scheduled game")
                    for snapshot in game["positions"]:
                        if snapshot["id"] in known:
                            continue
                        taught = teach(teacher, snapshot, args)
                        with data_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(taught, separators=(",", ":")) + "\n")
                            handle.flush()
                        rows.append(taught)
                        known.add(taught["id"])
                        print(f"  Teacher {taught['id']}: {taught.get('skipped') or taught['provenance']}, "
                              f"{taught['teacher_seconds']:.2f}s", flush=True)
                    progress["completed_games"][game_id] = game["summary"]
                    progress["pending"] = None
                    save_results(progress_path, progress)
            pairs, scale, metrics = fit(rows, args, previous=previous)
            metrics["training_decision"] = decide_training_stop(metrics, progress, args)
            metrics.update(cycle=cycle, updated_at=now(), source=str(args.source),
                           examples=len(rows), labelled_examples=sum(bool(row.get("target")) for row in rows),
                           completed_games=len(progress["completed_games"]),
                           configuration=configuration,
                           fit_settings={"epochs": args.epochs, "min_support": args.min_support,
                                         "max_entries": args.max_entries, "seed": args.fit_seed,
                                         "learning_rate": args.learning_rate,
                                         "lr_decay_epochs": args.lr_decay_epochs,
                                         "fit_patience": args.fit_patience,
                                         "fit_min_delta": args.fit_min_delta,
                                         "cycle_patience": args.cycle_patience,
                                         "cycle_min_delta": args.cycle_min_delta},
                           environment={"python": sys.version, "numpy": np.__version__,
                                        "platform": platform.platform()},
                           coefficients=[list(pair) for pair in pairs], exported=args.export)
            if args.export:
                export(args.source, pairs, scale)
            save_results(args.output_dir / "metrics.json", metrics)
            student.best_move._policy_data_override = pairs
            student.best_move._policy_scale_override = scale
            previous = metrics
            progress["training_decision"] = metrics["training_decision"]
            progress["next_cycle"] += 1
            save_results(progress_path, progress)
            print(f"Fit: {len(pairs)} entries, scale={scale}, "
                  f"validation CE {metrics['before']['validation']['cross_entropy']:.4f} -> "
                  f"{metrics['after']['validation']['cross_entropy']:.4f}; "
                  f"test CE {metrics['before']['test']['cross_entropy']:.4f} -> "
                  f"{metrics['after']['test']['cross_entropy']:.4f}", flush=True)
            print_training_status(metrics)
            if (metrics["training_decision"]["stop"] or
                    (args.forever and progress["next_cycle"] % args.evaluate_every == 0) or
                    (not args.forever and progress["next_cycle"] == end_cycle)):
                evaluate_checkpoint(args, configuration)
            if metrics["training_decision"]["stop"]:
                break
    except KeyboardInterrupt:
        save_results(progress_path, progress)
        print("Stopped; completed games and teacher examples are saved.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
