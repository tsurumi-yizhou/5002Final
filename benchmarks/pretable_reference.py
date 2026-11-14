import pygame as pg
import numpy as np
import time

IntGrid = np.ndarray[tuple[int, int], np.dtype[np.int_]]
EdgeGrid = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
State = tuple[IntGrid, EdgeGrid, EdgeGrid]
Point = tuple[int, int]
Analysis = tuple[int | None, list[int], float, bool]

# %% Constants that may be modified during tests ###############################

COLOR_EDGE: list[int] = [209, 139, 71]  # dark  brown
COLOR_BOARD: list[int] = [255, 206, 158]  # light brown
COLOR_STONE: dict[int, list[int]] = {
    +1: [0, 0, 0],  # black
    -1: [255, 255, 255],
}  # white
COLOR_BORDER: dict[int, list[int]] = {
    +1: [75, 75, 75],  # dark  grey
    -1: [235, 235, 235],
}  # light grey
COLOR_HIGHLIGHT: list[int] = [255, 0, 0]  # red

WINDOW_WIDTH: int = 640
WINDOW_HEIGHT: int = 720
WINDOW_CAPTION: str = "Five-in-a-row"

NUM_OF_ROWS: int = 17  # N_R
NUM_OF_COLS: int = 13  # N_C

PAD_x: float = WINDOW_WIDTH / NUM_OF_COLS
PAD_y: float = WINDOW_HEIGHT / NUM_OF_ROWS

SEP_x: float = (WINDOW_WIDTH - 2 * PAD_x) / (
    NUM_OF_COLS - 1
)  # separation between columns
SEP_y: float = (WINDOW_HEIGHT - 2 * PAD_y) / (
    NUM_OF_ROWS - 1
)  # separation between rows

STONE_RADIUS: float = min(SEP_x, SEP_y) * 0.3

CONNECTIVITY_HORIZONTAL: float = 0.85  # C_H = P(a horizontal edge exists)
CONNECTIVITY_VERTICAL: float = 0.7  # C_V = P(a vertical   edge exists)

# %% Utility functions that you can use ########################################


def draw_board(state: State) -> pg.Surface:
    """This function draws an empty board in the game window."""
    board, edgeH, edgeV = state

    surface = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    surface.fill(COLOR_BOARD)

    def draw_line(
        xy1: tuple[float, float], xy2: tuple[float, float], thickness: int
    ) -> None:
        pg.draw.line(surface, COLOR_EDGE, xy1, xy2, thickness)

    for r in range(NUM_OF_ROWS):
        for c in range(NUM_OF_COLS):
            x = PAD_x + c * SEP_x
            y = PAD_y + r * SEP_y
            pg.draw.circle(surface, COLOR_EDGE, (x, y), STONE_RADIUS / 2)
            if c > 0:
                draw_line((x - SEP_x, y), (x, y), 3 * int(edgeH[r, c - 1]))
            if r > 0:
                draw_line((x, y - SEP_y), (x, y), 3 * int(edgeV[r - 1, c]))

    pg.display.set_caption(WINDOW_CAPTION)
    pg.display.update()
    return surface


def draw_stone(
    surface: pg.Surface, r: int, c: int, color: int, highlight: bool = False
) -> None:
    """This function draws a stone in the rth row and cth column of
    the game board (0 <= r < NUM_OF_ROWS and 0 <= c < NUM_OF_COLS).

    If color = +1, the stone is black.
    If color = -1, the stone is white.
    If highlight = True, the stone has a highlighted border.
    """
    x = PAD_x + c * SEP_x
    y = PAD_y + r * SEP_y

    def draw_circle(color: list[int], width: int = 0) -> None:
        pg.draw.circle(surface, color, [x, y], STONE_RADIUS, width)

    draw_circle(COLOR_STONE[color])
    draw_circle(COLOR_BORDER[color], 2)
    if highlight:
        draw_circle(COLOR_HIGHLIGHT, 3)
    pg.display.update()


def print_result(surface: pg.Surface, winner: int) -> None:
    """The function prints the game result
    in the upper left corner of the game window.
    """
    if winner == 0:
        msg, c = "Draw!", COLOR_EDGE
    elif winner == +1:
        msg, c = "Black wins!", COLOR_STONE[+1]
    elif winner == -1:
        msg, c = "White wins!", COLOR_STONE[-1]
    else:
        raise ValueError("The result must be a draw, Black win, or White win.")

    font = pg.font.Font("freesansbold.ttf", 32)
    text = font.render(msg, True, c)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()


# %% Functions that you need to complete #######################################


def best_move(state: State, color: int) -> Point:
    """Combine exact-five threat search with an anytime, adversarial MCTS.

    Only windows affected by a move (including their overline boundaries) are
    updated. Forcing lines resolve checks before quiet play; MCTS progressively
    widens its legal move set and propagates proven outcomes. Rollout choices
    depend on the position, never on which tree children have been expanded.
    Easy decisions return immediately; uncertain decisions receive more time.
    Everything needed by the competition entry point is local to this call.
    """
    ###########################################################################
    started = time.perf_counter()
    hard_end = started + 4.5
    board, edgeH, edgeV = state
    rows, cols = board.shape
    size = rows * cols
    position = [int(value) for value in board.flat] + [0]
    empty = {point for point in range(size) if position[point] == 0}
    if not empty:
        raise ValueError("There are no empty points on the board.")
    if color not in (-1, +1):
        raise ValueError("The stone colour must be +1 or -1.")

    def connected(row: int, col: int, nr: int, nc: int) -> bool:
        if row == nr:
            return bool(edgeH[row, min(col, nc)])
        if col == nc:
            return bool(edgeV[min(row, nr), col])
        return True

    windows: list[tuple[int, ...]] = []
    before: list[int] = []
    after: list[int] = []
    interiors: list[list[int]] = [[] for _ in range(size)]
    affected: list[list[int]] = [[] for _ in range(size)]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(rows):
            for col in range(cols):
                pr, pc = row - dr, col - dc
                if 0 <= pr < rows and 0 <= pc < cols and connected(pr, pc, row, col):
                    continue
                line: list[int] = []
                nr, nc = row, col
                while 0 <= nr < rows and 0 <= nc < cols:
                    line.append(nr * cols + nc)
                    rr, cc = nr + dr, nc + dc
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        break
                    if not connected(nr, nc, rr, cc):
                        break
                    nr, nc = rr, cc
                for offset in range(len(line) - 4):
                    points = tuple(line[offset : offset + 5])
                    left = line[offset - 1] if offset else size
                    right = line[offset + 5] if offset + 5 < len(line) else size
                    wid = len(windows)
                    windows.append(points)
                    before.append(left)
                    after.append(right)
                    for point in points:
                        interiors[point].append(wid)
                        affected[point].append(wid)
                    if left != size:
                        affected[left].append(wid)
                    if right != size:
                        affected[right].append(wid)

    centrality = [
        1.0 / (1.0 + (point // cols - (rows - 1) / 2) ** 2
               + (point % cols - (cols - 1) / 2) ** 2)
        for point in range(size)
    ]
    if not windows:
        return divmod(max(empty, key=lambda point: (centrality[point], -point)), cols)

    count = [
        [sum(position[point] == player for point in points) for points in windows]
        for player in (+1, -1)
    ]
    empty_sum = [sum(point for point in points if position[point] == 0)
                 for points in windows]
    weights = (1, 6, 40, 320, 6000, 0)
    values = [[0] * len(windows) for _ in range(2)]
    totals = [0, 0]
    scores = [[0] * size for _ in range(2)]
    winning_count = [[0] * size for _ in range(2)]
    winning_points: list[set[int]] = [set(), set()]
    winning_window = [[-1] * len(windows) for _ in range(2)]
    threes: list[set[int]] = [set(), set()]
    fives = [set(), set()]

    def refresh(wid: int) -> None:
        left, right = position[before[wid]], position[after[wid]]
        for side, player in ((0, +1), (1, -1)):
            stones = count[side][wid]
            active = count[1 - side][wid] == 0 and left != player and right != player
            value = weights[stones] if active else 0
            difference = value - values[side][wid]
            if difference:
                values[side][wid] = value
                totals[side] += difference
                side_scores = scores[side]
                for point in windows[wid]:
                    side_scores[point] += difference
            old = winning_window[side][wid]
            new = empty_sum[wid] if active and stones == 4 else -1
            if old != new:
                side_counts = winning_count[side]
                if old >= 0:
                    side_counts[old] -= 1
                    if side_counts[old] == 0:
                        winning_points[side].discard(old)
                if new >= 0:
                    side_counts[new] += 1
                    winning_points[side].add(new)
                winning_window[side][wid] = new
            if active and stones == 3:
                threes[side].add(wid)
            else:
                threes[side].discard(wid)
            if active and stones == 5:
                fives[side].add(wid)
            else:
                fives[side].discard(wid)

    for wid in range(len(windows)):
        refresh(wid)

    def play(move: int, player: int) -> None:
        position[move] = player
        empty.remove(move)
        side_count = count[0 if player == +1 else 1]
        for wid in interiors[move]:
            side_count[wid] += 1
            empty_sum[wid] -= move
        for wid in affected[move]:
            refresh(wid)

    def undo(move: int) -> None:
        player = position[move]
        position[move] = 0
        empty.add(move)
        side_count = count[0 if player == +1 else 1]
        for wid in interiors[move]:
            side_count[wid] -= 1
            empty_sum[wid] += move
        for wid in affected[move]:
            refresh(wid)

    def forks(player: int) -> set[int]:
        """Moves creating two DISTINCT exact-five winning squares.

        An active three's two empty points form a threat pair: filling either
        leaves the other as a winning move. Two different partners prove a fork.
        A current opponent check must still be blocked by the fork move itself.
        """
        side = 0 if player == +1 else 1
        checks = winning_points[1 - side]
        if len(checks) > 1:
            return set()
        partners: dict[int, int] = {}
        result: set[int] = set()
        for wid in threes[side]:
            a, b = (point for point in windows[wid] if position[point] == 0)
            for point, other in ((a, b), (b, a)):
                if checks and point not in checks:
                    continue
                if point in partners and partners[point] != other:
                    result.add(point)
                else:
                    partners[point] = other
        return result

    def outcome(player: int) -> int:
        """Resolve actual wins, immediate wins and unavoidable double threats."""
        if fives[0]:
            return +1
        if fives[1] or not empty:
            return -1
        side = 0 if player == +1 else 1
        if winning_points[side]:
            return player
        if len(winning_points[1 - side]) >= 2:
            return -player
        if forks(player):
            return player
        # No remaining feasible black five means White can always secure a draw.
        if totals[0] == 0:
            return -1
        return 0

    def move_score(point: int, player: int) -> float:
        side = 0 if player == +1 else 1
        return scores[side][point] + 1.12 * scores[1 - side][point] + centrality[point]

    def ranked(player: int) -> list[int]:
        side = 0 if player == +1 else 1
        if winning_points[side]:
            choices = winning_points[side]
        elif len(winning_points[1 - side]) == 1:
            choices = winning_points[1 - side]
        else:
            choices = forks(player) or empty
        return sorted(choices, key=lambda point: (-move_score(point, player), point))

    root_side = 0 if color == +1 else 1
    candidates = ranked(color)
    if winning_points[root_side] or len(candidates) == 1 or outcome(color):
        return divmod(candidates[0], cols)
    if len(empty) == size:
        return divmod(candidates[0], cols)
    if len(empty) == size - 1:
        first = next(point for point in range(size) if position[point])
        neighbours = [point for point in candidates
                      if max(abs(point // cols - first // cols),
                             abs(point % cols - first % cols)) <= 1]
        if neighbours:
            return divmod(neighbours[0], cols)

    def forcing_line(attacker: int, depth: int, deadline: float) -> tuple[int, list[int]]:
        """Return 1 for a proven forcing win, 0 for no line at this depth,
        or -1 when the time budget expires. Every defending reply is forced.
        """
        side = 0 if attacker == +1 else 1
        result = outcome(attacker)
        if result:
            if result == attacker and winning_points[side]:
                return 1, [min(winning_points[side])]
            if result == attacker:
                winning_forks = forks(attacker)
                if winning_forks:
                    move = min(winning_forks)
                    partners = {point for wid in threes[side]
                                if move in windows[wid] for point in windows[wid]
                                if point != move and position[point] == 0}
                    return 1, [move] + sorted(partners)
            return (1 if result == attacker else 0), []
        if depth <= 0:
            return 0, []
        if time.perf_counter() >= deadline:
            return -1, []
        if winning_points[1 - side]:
            attacks = list(winning_points[1 - side])
        else:
            attacks = list({point for wid in threes[side] for point in windows[wid]
                            if position[point] == 0})
        attacks.sort(key=lambda point: (-move_score(point, attacker), point))
        for move in attacks:
            if time.perf_counter() >= deadline:
                return -1, []
            play(move, attacker)
            result = outcome(-attacker)
            if result:
                undo(move)
                if result == attacker:
                    return 1, [move]
                continue
            threats = winning_points[side]
            if not threats:
                undo(move)
                continue
            defence = min(threats)
            play(defence, -attacker)
            status, continuation = forcing_line(attacker, depth - 1, deadline)
            undo(defence)
            undo(move)
            if status == 1:
                return 1, [move, defence] + continuation
            if status == -1:
                return -1, []
        return 0, []

    attack_end = min(hard_end, time.perf_counter() + 0.07)
    status, line = forcing_line(color, 5, attack_end)
    if status == 1 and line:
        return divmod(line[0], cols)
    defence_end = min(hard_end, time.perf_counter() + 0.05)
    danger, enemy_line = forcing_line(-color, 4, defence_end)
    losing_moves: set[int] = set()
    screened_moves: set[int] = set()
    if danger == 1:
        # A defence may occupy any point on the forcing line or counterattack.
        critical = set(enemy_line)
        candidates.sort(key=lambda point: (point not in critical,
                                           -move_score(point, color), point))
        defence_end = min(hard_end, time.perf_counter() + 0.22)
        for move in candidates[:24]:
            if time.perf_counter() >= defence_end:
                break
            play(move, color)
            status, _ = forcing_line(-color, 4,
                                     min(defence_end, time.perf_counter() + 0.018))
            undo(move)
            if status == 1:
                losing_moves.add(move)
            elif status == 0:
                screened_moves.add(move)
        candidates = [move for move in candidates if move not in losing_moves]
        if not candidates:
            return divmod(ranked(color)[0], cols)
        if len(candidates) == 1:
            return divmod(candidates[0], cols)
        candidates.sort(key=lambda point: (point not in screened_moves,
                                           -move_score(point, color), point))

    class Node:
        __slots__ = ("move", "player", "moves", "children", "visits", "value", "solved")

        def __init__(self, move: int, player: int, moves: list[int] | None = None) -> None:
            self.move = move
            self.player = player
            self.solved = outcome(player)
            self.moves = moves if moves is not None else ([] if self.solved else ranked(player))
            self.children: list[Node] = []
            self.visits = 0
            self.value = 0.0

    root = Node(-1, color, candidates)
    complex_position = bool(threes[0] or threes[1] or danger == 1)
    soft_end = min(hard_end, max(started + 0.22, time.perf_counter() + 0.10))
    search_end = min(hard_end, started + (1.15 if danger == 1 else
                                        0.75 if complex_position else 0.50))
    gui_active = pg.get_init() and pg.display.get_init() and pg.display.get_surface() is not None
    next_event_check = time.perf_counter()
    stable_best = -1
    stable_checks = 0
    iterations = 0

    while time.perf_counter() < search_end:
        now = time.perf_counter()
        if gui_active and now >= next_event_check:
            pg.event.pump()
            if pg.event.peek(pg.QUIT):
                break
            next_event_check = now + 0.03
        node = root
        path = [root]
        played: list[int] = []
        while not node.solved:
            allowed = min(len(node.moves), 4 + int(node.visits ** 0.5))
            if len(node.children) < allowed:
                move = node.moves[len(node.children)]
                play(move, node.player)
                played.append(move)
                child = Node(move, -node.player)
                node.children.append(child)
                node = child
                path.append(node)
                break
            viable = [child for child in node.children if child.solved != -node.player]
            if not viable:
                if len(node.children) == len(node.moves):
                    node.solved = -node.player
                    break
                # Expand another legal move rather than sample a proven loss.
                move = node.moves[len(node.children)]
                play(move, node.player)
                played.append(move)
                child = Node(move, -node.player)
                node.children.append(child)
                node = child
                path.append(node)
                break
            logarithm = float(np.log(max(2, node.visits)))
            turn = node.player

            def uct(child: Node) -> float:
                mean = child.value / child.visits
                return (mean if turn == color else 1.0 - mean) + 0.75 * (
                    logarithm / child.visits
                ) ** 0.5

            child = max(viable, key=uct)
            play(child.move, node.player)
            played.append(child.move)
            node = child
            path.append(node)

        winner = node.solved
        player = node.player
        # Rollouts always use the current board's policy, including expanded moves.
        for step in range(24):
            if winner:
                break
            side = 0 if player == +1 else 1
            forcing = bool(winning_points[1 - side])
            if (step >= 10 and not forcing) or time.perf_counter() >= search_end:
                break
            if forcing:
                move = min(winning_points[1 - side])
            elif len(empty) > 1 and np.random.random() < 0.12:
                moves = ranked(player)
                move = moves[int(np.random.randint(min(3, len(moves))))]
            else:
                own_scores, other_scores = scores[side], scores[1 - side]
                move = max(empty, key=lambda point: (
                    own_scores[point] + 1.12 * other_scores[point] + centrality[point],
                    -point
                ))
            play(move, player)
            played.append(move)
            player = -player
            # Resolve a known win even when this was the final rollout move.
            winner = outcome(player)

        if winner:
            reward = float(winner == color)
        else:
            occupied_fraction = 1.0 - len(empty) / size
            advantage = totals[0] - totals[1] - 160.0 * occupied_fraction ** 4
            reward = float(0.5 + 0.45 * np.tanh(color * advantage / 1600.0))
        for visited in reversed(path):
            visited.visits += 1
            visited.value += reward
            if not visited.solved and visited.children:
                if any(child.solved == visited.player for child in visited.children):
                    visited.solved = visited.player
                elif len(visited.children) == len(visited.moves) and all(
                    child.solved == -visited.player for child in visited.children
                ):
                    visited.solved = -visited.player
        for move in reversed(played):
            undo(move)
        iterations += 1
        if root.solved:
            break
        if iterations % 16 == 0 and root.children:
            contenders = sorted(
                (child for child in root.children if child.solved != -color),
                key=lambda child: (child.visits, child.value / child.visits), reverse=True
            )
            if contenders:
                best = contenders[0]
                stable_checks = stable_checks + 1 if best.move == stable_best else 0
                stable_best = best.move
                if now >= soft_end and iterations >= 64 and stable_checks >= 3:
                    if len(contenders) == 1 or (
                        best.visits >= 1.5 * contenders[1].visits
                        and best.value / best.visits >=
                        contenders[1].value / contenders[1].visits + 0.04
                    ):
                        break

    if root.children:
        winners = [child for child in root.children if child.solved == color]
        viable = [child for child in root.children if child.solved != -color]
        choices = winners or viable
        if choices:
            chosen = max(choices, key=lambda child: (child.visits, child.value / child.visits))
            return divmod(chosen.move, cols)
        # Unexpanded legal moves are preferable to replies already proven to lose.
        if len(root.children) < len(root.moves):
            return divmod(root.moves[len(root.children)], cols)
    return divmod(candidates[0], cols)
    ###########################################################################



def check_result(state: State) -> tuple[bool, IntGrid]:
    """Given the current state=(board, edgeH, edgeV),
    this function checks if a player has already won the game.

    This function returns (gameover, winning_line).

    gameover: bool
        * True if a player has won or the game ends in a draw.
        * False if the game has not yet ended.

    winning_line: a 5*2 NumPy array
        * The positions (r, c) of the five stones that make a player win.
        * If no one wins (yet), its 10 elements are all -1.

    You can write between the two rows of #.
    """
    ###########################################################################
    board, edgeH, edgeV = state
    rows, cols = board.shape
    winning_line: IntGrid = np.full((5, 2), -1, dtype=np.int_)

    def connected(row: int, col: int, next_row: int, next_col: int) -> bool:
        if row == next_row:
            return bool(edgeH[row, min(col, next_col)])
        if col == next_col:
            return bool(edgeV[min(row, next_row), col])
        return True

    for row in range(rows):
        for col in range(cols):
            stone = board[row, col]
            if stone == 0:
                continue
            for delta_row, delta_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
                previous_row = row - delta_row
                previous_col = col - delta_col
                if (
                    0 <= previous_row < rows
                    and 0 <= previous_col < cols
                    and board[previous_row, previous_col] == stone
                    and connected(previous_row, previous_col, row, col)
                ):
                    continue
                line = [(row, col)]
                next_row, next_col = row, col
                while True:
                    following_row = next_row + delta_row
                    following_col = next_col + delta_col
                    if not (0 <= following_row < rows and 0 <= following_col < cols):
                        break
                    if board[following_row, following_col] != stone:
                        break
                    if not connected(next_row, next_col, following_row, following_col):
                        break
                    line.append((following_row, following_col))
                    if len(line) > 5:
                        break
                    next_row, next_col = following_row, following_col
                if len(line) == 5:
                    for index, point in enumerate(line):
                        winning_line[index] = point
                    return True, winning_line
    ###########################################################################
    return not bool(np.any(board == 0)), winning_line


def main(human_color: int) -> State:
    """Run a game and return its final board and connection arrays.

    The human plays Black for +1 and White for -1. Other values show computer
    self-play. Black always starts. Only valid left clicks during a human turn
    place stones; the latest move and the final winning line are highlighted.
    A finished game remains visible until the window is closed.
    """
    pg.init()
    board: IntGrid = np.zeros((NUM_OF_ROWS, NUM_OF_COLS), dtype=np.int_)

    random = np.random.random
    edgeH: EdgeGrid = np.empty((NUM_OF_ROWS, NUM_OF_COLS - 1), dtype=np.bool_)
    edgeV: EdgeGrid = np.empty((NUM_OF_ROWS - 1, NUM_OF_COLS), dtype=np.bool_)
    np.less(random(edgeH.shape), CONNECTIVITY_HORIZONTAL, out=edgeH)
    np.less(random(edgeV.shape), CONNECTIVITY_VERTICAL, out=edgeV)

    state: State = (board, edgeH, edgeV)
    surface = draw_board(state)

    running = True  # turns False once you try to close the game window
    gameover = False  # turns True once the game has ended
    congratulated = False  # turns True once the game result is printed

    previous_move: Point | None = None  # after the first move, it stores the position
    # (r, c) at which a stone has just been placed

    ###########################################################################

    turn = +1
    clock = pg.time.Clock()
    self_play = abs(human_color) != 1

    def place_stone(row: int, col: int) -> None:
        nonlocal turn, previous_move, gameover, congratulated
        if previous_move is not None:
            previous_row, previous_col = previous_move
            draw_stone(
                surface,
                previous_row,
                previous_col,
                int(board[previous_row, previous_col]),
                highlight=False,
            )
        board[row, col] = turn
        draw_stone(surface, row, col, turn, highlight=True)
        previous_move = (row, col)
        gameover, winning_line = check_result(state)
        if gameover and not congratulated:
            winner = 0
            if np.all(winning_line >= 0):
                first_row = int(winning_line[0, 0])
                first_col = int(winning_line[0, 1])
                winner = int(board[first_row, first_col])
                for winning_point in winning_line:
                    winning_row = int(winning_point[0])
                    winning_col = int(winning_point[1])
                    draw_stone(
                        surface, winning_row, winning_col, winner, highlight=True
                    )
            print_result(surface, winner)
            congratulated = True
        turn = -turn

    try:
        while running:
            moved = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    break
                if (
                    event.type == pg.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not gameover
                    and not self_play
                    and turn == human_color
                    and not moved
                ):
                    x, y = event.pos
                    row = round((y - PAD_y) / SEP_y)
                    col = round((x - PAD_x) / SEP_x)
                    if not (0 <= row < NUM_OF_ROWS and 0 <= col < NUM_OF_COLS):
                        continue
                    if (
                        abs(x - (PAD_x + col * SEP_x)) > SEP_x * 0.45
                        or abs(y - (PAD_y + row * SEP_y)) > SEP_y * 0.45
                        or board[row, col] != 0
                    ):
                        continue
                    place_stone(row, col)
                    moved = True

            if running and not gameover and (self_play or turn != human_color):
                row, col = best_move(state, turn)
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                if running:
                    place_stone(row, col)
            clock.tick(60)
    finally:
        pg.quit()

    ###########################################################################

    return state


if __name__ == "__main__":
    SEED = 5002  # used to initalize board; you can change it
    WINDOW_CAPTION += f" (seed: {SEED})"

    np.random.seed(SEED)
    state = main(human_color=+1)
