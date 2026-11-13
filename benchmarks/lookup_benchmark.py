"""Measure the actual lookup kernels on identical completed-game histories.

AST extraction keeps the best_move body prefix before the root_side assignment
unchanged, renames the enclosing function _benchmark_position, and appends
return locals(). Compilation uses a copy of the loaded module namespace, with
the production best_move global retained for its lazy immutable caches. There
is no runtime tracing or replacement of play, undo, outcome, or ranked.

PCG64 reconstructs each recorded edge layout. An untimed semantic replay
compares every forward and reverse position, including exact integer priorities
(100 * reference own score + 112 * reference opponent score), and verifies
restoration to each kernel's initial state. Ranked order is not compared:
scaling floating-point tie arithmetic can change ordering while integer
priorities agree. Tactical outcome can predict a win before a recorded game
ends, so replay always continues through the entire legal history.

Each timed batch repeats all games sequentially: play each recorded move, call
outcome for the next player, call ranked only when that outcome is zero, then
undo the whole game in reverse. Both engines perform the same operations;
engine order alternates by batch. Imports, AST extraction, board construction,
table/topology construction, and semantic comparisons are outside the clock.
Results measure these replay kernels, not initialization or full MCTS search.

Run from the project root, for example:
    python benchmarks/lookup_benchmark.py --output /tmp/lookup_results.json
"""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

if __package__:
    from .run import exact_five, load_agent, normalize_point, save_results
else:
    from run import exact_five, load_agent, normalize_point, save_results


ENGINES = ("reference", "candidate")
STATE_FIELDS = (
    "position", "empty", "count", "empty_sum", "totals",
    "winning_points", "threes", "fives",
)


def extract_position(path: Path, name: str):
    """Expose the original prefix's closures without altering their bodies."""
    module = load_agent(path, "lookup_" + name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = copy.deepcopy(next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "best_move"
    ))
    boundary = next(
        index for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "root_side"
                for target in node.targets)
    )
    function.name = "_benchmark_position"
    function.body = function.body[:boundary] + [
        ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()),
                                  args=[], keywords=[]))
    ]
    extracted = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = vars(module).copy()
    # Keep namespace['best_move'] pointing at the loaded production entry point.
    exec(compile(extracted, str(path), "exec", dont_inherit=True), namespace)
    return namespace[function.name]


def semantic_state(kernel, player):
    state = {key: kernel[key] for key in STATE_FIELDS}
    if "priorities" in kernel:
        state["priorities"] = kernel["priorities"]
    else:
        scores = kernel["scores"]
        state["priorities"] = [
            [100 * own + 112 * opposite
             for own, opposite in zip(scores[side], scores[1 - side])]
            for side in (0, 1)
        ]
    state["outcome"] = kernel["outcome"](player)
    return state


def compare_states(reference, candidate, context):
    for field in reference:
        if reference[field] != candidate[field]:
            raise ValueError(f"{context}: semantic mismatch in {field}")


def prepare_games(tournament, factories):
    """Rebuild completed legal histories and initialize both actual kernels."""
    settings = tournament["settings"]
    rows, cols = settings["rows"], settings["cols"]
    games = []
    for recorded in tournament["games"]:
        game_id = recorded["id"]
        if recorded.get("reason") not in ("exact_five", "full_board_white"):
            raise ValueError(f"Game {game_id} is not a completed legal game")
        rng = np.random.Generator(np.random.PCG64(recorded["seed"]))
        state = (
            np.zeros((rows, cols), dtype=np.int_),
            rng.random((rows, cols - 1)) < settings["horizontal"],
            rng.random((rows - 1, cols)) < settings["vertical"],
        )
        kernels = {name: factories[name](tuple(a.copy() for a in state), +1)
                   for name in ENGINES}
        if any(not isinstance(kernel, dict) for kernel in kernels.values()):
            raise ValueError(f"Game {game_id}: prefix returned before kernel setup")
        moves = []
        terminal = None
        for ply, move in enumerate(recorded["moves"], 1):
            player = +1 if ply % 2 else -1
            point = normalize_point(move["point"])
            if (terminal is not None or move["ply"] != ply
                    or move["color"] != player or not move["legal"]
                    or move.get("error") or point is None
                    or not (0 <= point[0] < rows and 0 <= point[1] < cols)
                    or state[0][tuple(point)] != 0):
                raise ValueError(f"Game {game_id}, ply {ply}: illegal recorded history")
            state[0][tuple(point)] = player
            moves.append((point[0] * cols + point[1], player))
            if exact_five(state, point, player):
                terminal = ("exact_five", player)
            elif ply == rows * cols:
                terminal = ("full_board_white", -1)
        if terminal != (recorded["reason"], recorded["winner_color"]):
            raise ValueError(f"Game {game_id}: recorded completion differs from replay")
        games.append({"id": game_id, "seed": recorded["seed"],
                      "moves": tuple(moves), "kernels": kernels})
    if not games:
        raise ValueError("No completed games to replay")
    return games


def semantic_replay(games):
    """Compare full trajectories and each engine's restoration, outside timing."""
    states_compared = 0
    ranked_calls = 0
    for game in games:
        kernels = game["kernels"]
        initial = {name: copy.deepcopy(semantic_state(kernels[name], +1))
                   for name in ENGINES}
        compare_states(initial["reference"], initial["candidate"],
                       f"Game {game['id']}, initial")
        states_compared += 1
        for ply, (point, player) in enumerate(game["moves"], 1):
            for name in ENGINES:
                kernels[name]["play"](point, player)
            states = {name: semantic_state(kernels[name], -player) for name in ENGINES}
            compare_states(states["reference"], states["candidate"],
                           f"Game {game['id']}, forward ply {ply}")
            states_compared += 1
            ranked_calls += int(states["reference"]["outcome"] == 0)
        for ply in range(len(game["moves"]), 0, -1):
            point, player = game["moves"][ply - 1]
            for name in ENGINES:
                kernels[name]["undo"](point)
            states = {name: semantic_state(kernels[name], player) for name in ENGINES}
            compare_states(states["reference"], states["candidate"],
                           f"Game {game['id']}, undo ply {ply}")
            states_compared += 1
        for name in ENGINES:
            compare_states(initial[name], semantic_state(kernels[name], +1),
                           f"Game {game['id']}, {name} restored initial state")
    return {"valid": True, "states_compared": states_compared,
            "games_restored": len(games), "ranked_calls_per_replay": ranked_calls,
            "compared_fields": [*STATE_FIELDS, "priorities", "outcome"]}


def timed_replays(tasks, replays):
    """Use the same driver and complete move histories for either engine."""
    ranked_calls = 0
    started = time.perf_counter()
    for _ in range(replays):
        for play, undo, outcome, ranked, moves in tasks:
            for point, player in moves:
                play(point, player)
                if outcome(-player) == 0:
                    ranked(-player)
                    ranked_calls += 1
            for point, _ in reversed(moves):
                undo(point)
    elapsed = time.perf_counter() - started
    return elapsed, ranked_calls


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path,
                        default=Path("results/benchmarks/pretable_tournament.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("benchmarks/pretable_reference.py"))
    parser.add_argument("--candidate", type=Path, default=Path("gomoku.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=positive_int, default=5)
    parser.add_argument("--replays", type=positive_int, default=50)
    args = parser.parse_args()
    if args.output.resolve() in {
        args.games.resolve(), args.reference.resolve(), args.candidate.resolve(),
        Path(__file__).resolve(), Path(__file__).with_name("run.py").resolve(),
    }:
        parser.error("--output must be separate from benchmark inputs and source files")
    results = {
        "measurement_type": "same_work_lookup_kernel_replay",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "valid": False,
        "sources": {"games": str(args.games.resolve()),
                    "reference": str(args.reference.resolve()),
                    "candidate": str(args.candidate.resolve())},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "processor": platform.processor(),
                        "clock": vars(time.get_clock_info("perf_counter"))},
        "settings": {"batches": args.batches, "replays_per_batch": args.replays,
                     "parallel_calls": False,
                     "order": "reference first on odd batches; candidate first on even batches",
                     "timed_work": "play, outcome(next player), ranked if outcome zero; reverse undo",
                     "excluded_work": "imports, extraction, initialization, semantic comparisons"},
        "batches": [],
    }
    try:
        tournament = json.loads(args.games.read_text(encoding="utf-8"))
        factories = {name: extract_position(getattr(args, name), name) for name in ENGINES}
        games = prepare_games(tournament, factories)
        results["board_settings"] = tournament["settings"]
        results["games"] = [{"id": g["id"], "seed": g["seed"],
                             "moves": len(g["moves"])} for g in games]
        results["semantic_replay"] = semantic_replay(games)
        moves = sum(len(game["moves"]) for game in games) * args.replays
        work = {"game_replays": len(games) * args.replays,
                "play": moves, "outcome": moves, "undo": moves,
                "ranked": results["semantic_replay"]["ranked_calls_per_replay"] * args.replays}
        tasks = {name: [tuple(game["kernels"][name][key]
                             for key in ("play", "undo", "outcome", "ranked"))
                        + (game["moves"],) for game in games] for name in ENGINES}
        for index in range(args.batches):
            order = ENGINES if index % 2 == 0 else ENGINES[::-1]
            batch = {"batch": index + 1, "order": list(order), "engines": {}}
            results["batches"].append(batch)
            for name in order:
                elapsed, ranked_calls = timed_replays(tasks[name], args.replays)
                actual_work = {**work, "ranked": ranked_calls}
                batch["engines"][name] = {"elapsed_seconds": elapsed,
                                          "work_count": actual_work}
                if actual_work != work:
                    raise ValueError(f"Batch {index + 1}, {name}: replay work differs")
            seconds = {name: batch["engines"][name]["elapsed_seconds"] for name in ENGINES}
            batch["ratio_reference_over_candidate"] = seconds["reference"] / seconds["candidate"]
            print(f"Batch {index + 1}/{args.batches}: reference={seconds['reference']:.6f}s, "
                  f"candidate={seconds['candidate']:.6f}s, "
                  f"ratio={batch['ratio_reference_over_candidate']:.3f}", flush=True)
        medians = {name: statistics.median(
            batch["engines"][name]["elapsed_seconds"] for batch in results["batches"]
        ) for name in ENGINES}
        results["summary"] = {
            "median_elapsed_seconds": medians,
            "ratio_of_median_times_reference_over_candidate": medians["reference"] / medians["candidate"],
            "median_batch_ratio_reference_over_candidate": statistics.median(
                batch["ratio_reference_over_candidate"] for batch in results["batches"]),
            "ratio_interpretation": "greater than one means candidate is faster",
            "work_count_per_engine_per_batch": work,
            "states_compared": results["semantic_replay"]["states_compared"],
        }
        results["valid"] = True
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(results["error"], file=sys.stderr)
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(args.output, results)
    print(f"Saved {args.output}; valid={results['valid']}", flush=True)
    return 0 if results["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
