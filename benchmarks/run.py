"""Sequential, headless full-game comparison against the frozen original.

Each seed supplies one independently generated edge layout, played in both roles.
PCG64 generates boards; each call reseeds NumPy's global RNG from seed, ply,
colour and agent identity. Wall-clock search budgets can still vary the moves.
Only best_move is timed: private input copies, RNG setup and mutation/legality
checks are outside the clock. Exceptions, mutated inputs, illegal moves and
calls exceeding five seconds forfeit the round. Unix SIGALRM interrupts slow
calls; elsewhere the elapsed limit is enforced on return.

The independent arbiter counts maximal connected runs through the last move;
only exactly five wins, diagonals are always connected, and a full board goes
to White after checking wins. Consecutive distinct seeds form two cross-colour
two-round matches: [seed A Black, seed B White] and the reverse. Split matches
go to the lower total thinking time; exact time ties remain ties. Round wins
earn one point and a two-round sweep earns one additional point. Unpaired seeds
still count as raw rounds. JSON is atomically replaced after every full game;
progress is flushed every ten calls and at game boundaries. No GUI is opened.

Alternatively --positions-from replays a tournament's legal history and times
both engines on identical nonterminal positions before recorded plies >= 3.
Samples are evenly spaced in stored game/move order, independent of timings or
outcomes. Each repeat resets the same RNG seed for both engines and alternates
their order; private input copies keep the position fixed. All calls, including
invalid results and timeouts, enter timing statistics (linear p95). These
measurements are labelled separately from head-to-head tournament timings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time

import numpy as np


MOVE_LIMIT = 5.0
AGENTS = ("candidate", "baseline")


class MoveTimeout(TimeoutError):
    pass


def load_agent(path: Path, name: str):
    """Execute each source once, without running its interactive main block."""
    spec = importlib.util.spec_from_file_location("benchmark_" + name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "best_move", None)):
        raise ValueError(f"{path} does not define best_move")
    return module


def timed_move(function, state, color):
    """Return point, elapsed seconds and error; always disarm the timer."""
    def expire(signum, frame):
        raise MoveTimeout("best_move exceeded five seconds")

    use_alarm = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
    previous = signal.signal(signal.SIGALRM, expire) if use_alarm else None
    point, error = None, None
    try:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, MOVE_LIMIT)
        started = time.perf_counter()
        try:
            point = function(state, color)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            seconds = time.perf_counter() - started
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    return point, seconds, error


def exact_five(state, point, color):
    """Count both rays in full, including sixth stones and broken H/V edges."""
    board, horizontal, vertical = state
    rows, cols = board.shape
    row, col = point
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        length = 1
        for sign in (-1, 1):
            r, c = row, col
            while True:
                nr, nc = r + sign * dr, c + sign * dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if board[nr, nc] != color:
                    break
                if dr == 0 and not horizontal[r, min(c, nc)]:
                    break
                if dc == 0 and not vertical[min(r, nr), c]:
                    break
                length += 1
                r, c = nr, nc
        if length == 5:
            return True
    return False


def normalize_point(value):
    if not isinstance(value, (tuple, list, np.ndarray)):
        return None
    try:
        if len(value) != 2:
            return None
        if any(isinstance(x, (bool, np.bool_)) or
               not isinstance(x, (int, np.integer)) for x in value):
            return None
        return [int(value[0]), int(value[1])]
    except TypeError:
        return None


def play_game(modules, settings, seed, candidate_color, game_id):
    rows, cols = settings["rows"], settings["cols"]
    rng = np.random.Generator(np.random.PCG64(seed))
    state = (
        np.zeros((rows, cols), dtype=np.int_),
        rng.random((rows, cols - 1)) < settings["horizontal"],
        rng.random((rows - 1, cols)) < settings["vertical"],
    )
    game = {
        "id": game_id, "seed": seed, "candidate_color": candidate_color,
        "moves": [], "seconds": dict.fromkeys(AGENTS, 0.0),
        "move_max_seconds": dict.fromkeys(AGENTS, 0.0),
        "checks": {name: {"calls": 0, "legal_moves": 0,
                          "illegal_moves": 0, "input_mutations": 0,
                          "over_five_seconds": 0, "exceptions": 0}
                   for name in AGENTS},
    }
    print(f"Game {game_id}: seed={seed}, candidate "
          f"{'Black' if candidate_color == 1 else 'White'}", flush=True)
    for ply in range(1, rows * cols + 1):
        color = 1 if ply % 2 else -1
        name = "candidate" if color == candidate_color else "baseline"
        # Agent mutations cannot alter the tournament position.
        supplied = tuple(array.copy() for array in state)
        agent_seed = int(np.random.SeedSequence(
            [seed, ply, int(color == 1), AGENTS.index(name)]
        ).generate_state(1)[0])
        np.random.seed(agent_seed)
        value, seconds, error = timed_move(modules[name].best_move, supplied, color)
        point = normalize_point(value)
        mutated = any(a.shape != b.shape or a.dtype != b.dtype or
                      not np.array_equal(a, b) for a, b in zip(supplied, state))
        legal = (point is not None and 0 <= point[0] < rows and
                 0 <= point[1] < cols and state[0][tuple(point)] == 0)
        late = seconds > MOVE_LIMIT or (error is not None and
                                       error.startswith("MoveTimeout:"))
        counts = game["checks"][name]
        counts["calls"] += 1
        counts["input_mutations"] += int(mutated)
        counts["over_five_seconds"] += int(late)
        counts["exceptions"] += int(error is not None and
                                    not error.startswith("MoveTimeout:"))
        # No returned move on a timeout/exception is recorded as an illegal move.
        if error is None:
            counts["legal_moves" if legal else "illegal_moves"] += 1
        game["seconds"][name] += seconds
        game["move_max_seconds"][name] = max(game["move_max_seconds"][name], seconds)
        move = {"ply": ply, "agent": name, "color": color,
                "point": point, "seconds": seconds, "rng_seed": agent_seed,
                "legal": bool(legal), "input_unchanged": not mutated,
                "within_five_seconds": not late}
        if error:
            move["error"] = error
        game["moves"].append(move)
        violations = []
        if error:
            violations.append(error)
        if late:
            violations.append("move_time_limit")
        if mutated:
            violations.append("input_mutation")
        if error is None and not legal:
            violations.append("illegal_move")
        if ply % 10 == 0:
            print(f"  game {game_id}, ply {ply}: candidate="
                  f"{game['seconds']['candidate']:.3f}s, baseline="
                  f"{game['seconds']['baseline']:.3f}s", flush=True)
        if violations:
            game.update(winner_color=-color, reason="forfeit",
                        forfeiting_agent=name, violations=violations)
            break
        state[0][tuple(point)] = color
        if exact_five(state, point, color):
            game.update(winner_color=color, reason="exact_five")
            break
        if ply == rows * cols:
            game.update(winner_color=-1, reason="full_board_white")
            break
    game["winner"] = ("candidate" if game["winner_color"] == candidate_color
                      else "baseline")
    return game


def summarize(games, seeds):
    by_role = {role: {"games": 0, "wins": 0} for role in ("black", "white")}
    totals = {name: {"seconds": 0.0, "calls": 0, "move_max_seconds": 0.0,
                     "legal_moves": 0, "illegal_moves": 0, "input_mutations": 0,
                     "over_five_seconds": 0, "exceptions": 0} for name in AGENTS}
    for game in games:
        role = by_role["black" if game["candidate_color"] == 1 else "white"]
        role["games"] += 1
        role["wins"] += int(game["winner"] == "candidate")
        for name in AGENTS:
            total = totals[name]
            total["seconds"] += game["seconds"][name]
            total["move_max_seconds"] = max(total["move_max_seconds"],
                                             game["move_max_seconds"][name])
            for key, count in game["checks"][name].items():
                total[key] += count
    for total in totals.values():
        total["mean_seconds_per_move"] = (total["seconds"] / total["calls"]
                                            if total["calls"] else None)
    lookup = {(g["seed"], g["candidate_color"]): g for g in games}
    matches = []
    for offset in range(0, len(seeds) - 1, 2):
        first, second = seeds[offset:offset + 2]
        for color in (1, -1):
            rounds = [lookup.get((first, color)), lookup.get((second, -color))]
            if any(g is None for g in rounds):
                continue
            wins = {name: sum(g["winner"] == name for g in rounds) for name in AGENTS}
            seconds = {name: sum(g["seconds"][name] for g in rounds) for name in AGENTS}
            winner, decision = None, "exact_time_tie"
            if wins["candidate"] != wins["baseline"]:
                winner = max(AGENTS, key=wins.get)
                decision = "round_wins"
            elif seconds["candidate"] != seconds["baseline"]:
                winner = min(AGENTS, key=seconds.get)
                decision = "total_thinking_time"
            matches.append({"seeds": [first, second],
                            "game_ids": [g["id"] for g in rounds],
                            "candidate_first_color": color, "round_wins": wins,
                            "seconds": seconds, "winner": winner,
                            "decision": decision,
                            "points": {n: wins[n] + int(wins[n] == 2) for n in AGENTS}})
    candidate_mean = totals["candidate"]["mean_seconds_per_move"]
    baseline_mean = totals["baseline"]["mean_seconds_per_move"]
    return {
        "completed_games": len(games), "planned_games": 2 * len(seeds),
        "candidate_raw_round_wins": sum(g["winner"] == "candidate" for g in games),
        "candidate_by_color": by_role, "agents": totals,
        "speed_ratio_baseline_over_candidate_per_move": (
            baseline_mean / candidate_mean if candidate_mean and baseline_mean is not None
            else None),
        "official_matches": matches,
        "official_match_wins": {n: sum(m["winner"] == n for m in matches) for n in AGENTS},
        "official_match_ties": sum(m["winner"] is None for m in matches),
        "official_points": {n: sum(m["points"][n] for m in matches) for n in AGENTS},
        "unpaired_seeds": seeds[-1:] if len(seeds) % 2 else [],
    }


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(results, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replay_positions(tournament):
    """Yield pre-move states; callers must copy arrays they retain across yields."""
    settings = tournament["settings"]
    rows, cols = settings["rows"], settings["cols"]
    for game_index, game in enumerate(tournament["games"]):
        rng = np.random.Generator(np.random.PCG64(game["seed"]))
        state = (
            np.zeros((rows, cols), dtype=np.int_),
            rng.random((rows, cols - 1)) < settings["horizontal"],
            rng.random((rows - 1, cols)) < settings["vertical"],
        )
        for ply, move in enumerate(game["moves"], 1):
            color = 1 if ply % 2 else -1
            if move["ply"] != ply or move["color"] != color:
                raise ValueError(f"Inconsistent history in game {game['id']}, ply {ply}")
            # A forfeited attempt was never placed on the tournament board.
            if (move.get("error") or not move["legal"] or
                    not move["input_unchanged"] or not move["within_five_seconds"]):
                break
            point = normalize_point(move["point"])
            if (point is None or not (0 <= point[0] < rows and 0 <= point[1] < cols)
                    or state[0][tuple(point)] != 0):
                raise ValueError(f"Illegal history in game {game['id']}, ply {ply}")
            if ply >= 3:
                yield {"game_index": game_index, "game_id": game["id"],
                       "seed": game["seed"], "ply": ply, "color": color}, state
            state[0][tuple(point)] = color
            if exact_five(state, point, color) or ply == rows * cols:
                break


def summarize_positions(samples):
    agents = {}
    for name in AGENTS:
        calls = [call for sample in samples for call in sample["calls"]
                 if call["agent"] == name]
        seconds = [call["seconds"] for call in calls]
        agents[name] = {
            "calls": len(calls), "valid_calls": sum(call["valid"] for call in calls),
            "invalid_calls": sum(not call["valid"] for call in calls),
            "total_seconds": float(sum(seconds)),
            "mean_seconds": float(np.mean(seconds)) if seconds else None,
            "median_seconds": float(np.median(seconds)) if seconds else None,
            "p95_seconds": float(np.percentile(seconds, 95)) if seconds else None,
            "move_max_seconds": max(seconds) if seconds else None,
            "illegal_moves": sum(not call["legal"] and "error" not in call for call in calls),
            "input_mutations": sum(not call["input_unchanged"] for call in calls),
            "over_five_seconds": sum(not call["within_five_seconds"] for call in calls),
            "errors": sum("error" in call for call in calls),
        }
    candidate = agents["candidate"]["mean_seconds"]
    baseline = agents["baseline"]["mean_seconds"]
    return {"timing_source": "matched_position_timing",
            "completed_samples": len(samples), "agents": agents,
            "statistics_population": "all calls, including invalid results and timeouts",
            "p95_method": "linear interpolation at 0.95 * (call_count - 1)",
            "speed_ratio_baseline_over_candidate_per_move": (
                baseline / candidate if candidate and baseline is not None else None)}


def measure_positions(args, modules, results):
    with args.positions_from.open(encoding="utf-8") as handle:
        tournament = json.load(handle)
    # First pass only counts eligible states; no engine is consulted in selection.
    count = sum(1 for _ in replay_positions(tournament))
    if not count:
        raise ValueError("Source tournament has no eligible positions before legal plies >= 3")
    size = min(args.samples, count)
    indices = ([(count - 1) // 2] if size == 1 else
               [i * (count - 1) // (size - 1) for i in range(size)])
    results.pop("games")
    results.update(
        measurement_type="matched_position_timing",
        source_tournament={"path": str(args.positions_from),
                           "timing_source": "head_to_head_tournament",
                           "started_at": tournament.get("started_at"),
                           "updated_at": tournament.get("updated_at"),
                           "sources": tournament.get("sources"),
                           "settings": tournament["settings"],
                           "completed_games": len(tournament["games"])},
        settings={
            "samples_requested": args.samples, "samples_selected": size,
            "repeats": args.repeats, "move_limit_seconds": MOVE_LIMIT,
            "parallel_calls": False,
            "agent_rng": "SeedSequence([seed, game_index, ply, repeat]) -> np.random.seed; same for both engines",
            "call_order": "candidate first when (sample_index + repeat - 1) is even; baseline first otherwise",
            "timeout_enforcement": results["settings"]["timeout_enforcement"],
            "selection": {
                "population": "pre-move nonterminal states of legal, unchanged, on-time recorded moves with ply >= 3; stored game order then move order",
                "eligible_positions": count, "indices_zero_based": indices,
                "rule": "k=min(samples,N); floor(i*(N-1)/(k-1)) for i=0..k-1; k=1 uses floor((N-1)/2)",
                "uses_timings_or_outcomes": False,
            },
        },
        samples=[],
    )
    selected = set(indices)
    for population_index, (metadata, state) in enumerate(replay_positions(tournament)):
        if population_index not in selected:
            continue
        sample_index = len(results["samples"])
        sample = {**metadata, "sample_index": sample_index,
                  "population_index": population_index, "calls": []}
        color = metadata["color"]
        rows, cols = state[0].shape
        print(f"Sample {sample_index + 1}/{size}: game={metadata['game_id']}, "
              f"ply={metadata['ply']}, color={color:+d}", flush=True)
        for repeat in range(1, args.repeats + 1):
            rng_seed = int(np.random.SeedSequence([
                metadata["seed"], metadata["game_index"], metadata["ply"], repeat
            ]).generate_state(1)[0])
            order = AGENTS if (sample_index + repeat - 1) % 2 == 0 else AGENTS[::-1]
            for call_order, name in enumerate(order, 1):
                supplied = tuple(array.copy() for array in state)
                np.random.seed(rng_seed)
                value, seconds, error = timed_move(modules[name].best_move, supplied, color)
                point = normalize_point(value)
                unchanged = all(a.shape == b.shape and a.dtype == b.dtype and
                                np.array_equal(a, b) for a, b in zip(supplied, state))
                legal = (point is not None and 0 <= point[0] < rows and
                         0 <= point[1] < cols and state[0][tuple(point)] == 0)
                within_limit = seconds <= MOVE_LIMIT and not (
                    error is not None and error.startswith("MoveTimeout:"))
                call = {"agent": name, "repeat": repeat, "order": call_order,
                        "rng_seed": rng_seed, "seconds": seconds, "point": point,
                        "legal": bool(legal), "input_unchanged": unchanged,
                        "within_five_seconds": within_limit,
                        "valid": bool(legal and unchanged and within_limit and error is None)}
                if error:
                    call["error"] = error
                sample["calls"].append(call)
                print(f"  repeat {repeat}, {name}: {seconds:.6f}s, "
                      f"point={point}, valid={call['valid']}", flush=True)
        results["samples"].append(sample)
        results["summary"] = summarize_positions(results["samples"])
        results["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_results(args.output, results)
        print(f"Sample {sample_index + 1} complete; saved {args.output}", flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seeds", type=int, nargs="+")
    mode.add_argument("--positions-from", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="retain completed games in output; use unchanged agent sources")
    parser.add_argument("--candidate", type=Path, default=root / "gomoku.py")
    parser.add_argument("--baseline", type=Path, default=root / "benchmarks/baseline.py")
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--cols", type=int, default=13)
    parser.add_argument("--horizontal", type=float, default=0.85)
    parser.add_argument("--vertical", type=float, default=0.7)
    args = parser.parse_args()
    if args.resume and args.positions_from:
        parser.error("--resume applies to complete-game tournaments")
    if args.seeds is not None and (any(s < 0 for s in args.seeds) or
                                   len(set(args.seeds)) != len(args.seeds)):
        parser.error("seeds must be distinct nonnegative integers")
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    if args.rows < 1 or args.cols < 1:
        parser.error("rows and cols must be positive")
    if not (0 <= args.horizontal <= 1 and 0 <= args.vertical <= 1):
        parser.error("connectivities must lie in [0, 1]")
    args.candidate, args.baseline = args.candidate.resolve(), args.baseline.resolve()
    args.output = args.output.resolve()
    if args.output in (args.candidate, args.baseline, Path(__file__).resolve()):
        parser.error("output must not overwrite runner or agent source")
    if args.positions_from is not None:
        args.positions_from = args.positions_from.resolve()
        if args.output == args.positions_from:
            parser.error("output must be separate from the source tournament")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame

    modules = {"candidate": load_agent(args.candidate, "candidate"),
               "baseline": load_agent(args.baseline, "baseline")}
    settings = {"seeds": args.seeds, "rows": args.rows, "cols": args.cols,
                "horizontal": args.horizontal, "vertical": args.vertical,
                "move_limit_seconds": MOVE_LIMIT, "parallel_games": False,
                "board_rng": "numpy.Generator(PCG64(seed))",
                "agent_rng": "SeedSequence([seed, ply, color == 1, agent_index]) -> np.random.seed",
                "agent_indices": dict(zip(AGENTS, range(len(AGENTS)))),
                "timeout_enforcement": ("SIGALRM and elapsed time" if hasattr(signal, "setitimer")
                                        else "elapsed time on return"),
                "match_tiebreak": "lower combined thinking time across both rounds"}
    results = {"measurement_type": "head_to_head_tournament",
               "started_at": datetime.now(timezone.utc).isoformat(),
               "environment": {"python": sys.version, "numpy": np.__version__,
                               "pygame": pygame.version.ver, "platform": platform.platform(),
                               "executable": sys.executable},
               "sources": {"candidate": str(args.candidate), "baseline": str(args.baseline)},
               "settings": settings, "games": []}
    if args.positions_from is not None:
        measure_positions(args, modules, results)
        return
    if args.resume:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["settings"] != settings or previous["sources"] != results["sources"]:
            parser.error("resume requires the saved tournament's settings and source paths")
        results = previous
        results.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat())
    completed = {(game["seed"], game["candidate_color"]) for game in results["games"]}
    for seed in args.seeds:
        for color in (1, -1):
            if (seed, color) in completed:
                continue
            game = play_game(modules, settings, seed, color, len(results["games"]) + 1)
            results["games"].append(game)
            results["summary"] = summarize(results["games"], args.seeds)
            results["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_results(args.output, results)
            print(f"Game {game['id']} complete: {game['winner']} wins "
                  f"({game['reason']}), {len(game['moves'])} calls; "
                  f"candidate={game['seconds']['candidate']:.3f}s, "
                  f"baseline={game['seconds']['baseline']:.3f}s. "
                  f"Saved {args.output}", flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
