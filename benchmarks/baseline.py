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
    """Choose a legal move using tactical rules and Monte Carlo tree search.

    Connected five-point windows and their neighbours encode exact-five wins.
    Immediate wins, forced blocks and double threats precede UCT search.
    Search alternates the players' objectives, expands ranked moves, performs
    tactical rollouts and backs up rewards from the root player's perspective.
    Rollouts stop after sixteen moves and use a bounded window evaluation.
    Full-board draws reward White, following the competition rules.

    The total search budget is 1.8 seconds. All search data is local and the
    supplied board and edges remain unchanged. Colours are +1 for Black and
    -1 for White; the returned point contains Python integer coordinates.
    """
    ###########################################################################

    started = time.perf_counter()
    deadline = started + 1.8
    board, edgeH, edgeV = state
    rows, cols = board.shape
    size = rows * cols
    position = np.zeros(size + 1, dtype=np.int8)
    position[:size] = board.reshape(-1)
    empty = np.flatnonzero(position[:size] == 0)
    if not len(empty):
        raise ValueError("There are no empty points on the board.")
    if color not in (-1, +1):
        raise ValueError("The stone colour must be +1 or -1.")

    def connected(row: int, col: int, next_row: int, next_col: int) -> bool:
        if row == next_row:
            return bool(edgeH[row, min(col, next_col)])
        if col == next_col:
            return bool(edgeV[min(row, next_row), col])
        return True

    window_points: list[list[int]] = []
    preceding_points: list[int] = []
    following_points: list[int] = []
    for delta_row, delta_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(rows):
            for col in range(cols):
                previous_row = row - delta_row
                previous_col = col - delta_col
                if (
                    0 <= previous_row < rows
                    and 0 <= previous_col < cols
                    and connected(previous_row, previous_col, row, col)
                ):
                    continue
                line: list[int] = []
                next_row, next_col = row, col
                while 0 <= next_row < rows and 0 <= next_col < cols:
                    line.append(next_row * cols + next_col)
                    following_row = next_row + delta_row
                    following_col = next_col + delta_col
                    if not (0 <= following_row < rows and 0 <= following_col < cols):
                        break
                    if not connected(next_row, next_col, following_row, following_col):
                        break
                    next_row, next_col = following_row, following_col
                for offset in range(len(line) - 4):
                    window_points.append(line[offset : offset + 5])
                    preceding_points.append(line[offset - 1] if offset else size)
                    following_points.append(
                        line[offset + 5] if offset + 5 < len(line) else size
                    )

    if not window_points:
        distances = (empty // cols - (rows - 1) / 2) ** 2 + (
            empty % cols - (cols - 1) / 2
        ) ** 2
        return divmod(int(empty[np.argmin(distances)]), cols)

    windows = np.asarray(window_points, dtype=int)
    before = np.asarray(preceding_points, dtype=int)
    after = np.asarray(following_points, dtype=int)
    flattened_windows = windows.reshape(-1)
    weights = np.array([0.2, 2.0, 14.0, 110.0, 2400.0, 0.0])
    indices = np.arange(size)
    centrality = 1.0 / (
        1.0
        + (indices // cols - (rows - 1) / 2) ** 2
        + (indices % cols - (cols - 1) / 2) ** 2
    )

    def analyse(player: int, width: int) -> Analysis:
        stones = position[windows]
        black_count = np.count_nonzero(stones == +1, axis=1)
        white_count = np.count_nonzero(stones == -1, axis=1)
        black_open = (
            (white_count == 0) & (position[before] != +1) & (position[after] != +1)
        )
        white_open = (
            (black_count == 0) & (position[before] != -1) & (position[after] != -1)
        )
        if np.any(black_open & (black_count == 5)):
            return +1, [], 0.0, False
        if np.any(white_open & (white_count == 5)):
            return -1, [], 0.0, False
        available = np.flatnonzero(position[:size] == 0)
        if not len(available):
            return -1, [], 0.0, False

        black_weight = weights[black_count] * black_open
        white_weight = weights[white_count] * white_open
        advantage = float(black_weight.sum() - white_weight.sum())
        if player == +1:
            own_count, own_open = black_count, black_open
            other_count, other_open = white_count, white_open
            move_weight = black_weight + 1.1 * white_weight
        else:
            own_count, own_open = white_count, white_open
            other_count, other_open = black_count, black_open
            move_weight = white_weight + 1.1 * black_weight

        scores = (
            np.bincount(
                flattened_windows,
                weights=np.repeat(move_weight, 5),
                minlength=size,
            )
            + centrality
        )
        winning_windows = windows[own_open & (own_count == 4)]
        immediate = bool(len(winning_windows))
        if immediate:
            candidates = np.unique(winning_windows[position[winning_windows] == 0])
        else:
            blocking_windows = windows[other_open & (other_count == 4)]
            if len(blocking_windows):
                candidates = np.unique(
                    blocking_windows[position[blocking_windows] == 0]
                )
            else:
                candidates = available
        ordering = np.argsort(-scores[candidates], kind="stable")[:width]
        return None, [int(move) for move in candidates[ordering]], advantage, immediate

    winner, candidates, advantage, immediate = analyse(color, 18)
    if winner is not None:
        return divmod(int(empty[0]), cols)
    if immediate or len(candidates) == 1:
        return divmod(candidates[0], cols)
    if len(empty) == size:
        return divmod(candidates[0], cols)

    for move in candidates[:10]:
        if time.perf_counter() >= deadline:
            break
        position[move] = color
        _, _, _, opponent_can_win = analyse(-color, size)
        _, threats, _, creates_threat = analyse(color, size)
        position[move] = 0
        if creates_threat and len(threats) >= 2 and not opponent_can_win:
            return divmod(move, cols)

    class Node:
        __slots__ = (
            "move",
            "player",
            "children",
            "untried",
            "visits",
            "value",
            "winner",
            "advantage",
            "immediate",
        )

        def __init__(self, move: int, player: int, result: Analysis) -> None:
            self.move = move
            self.player = player
            self.children: list[Node] = []
            self.winner, moves, self.advantage, self.immediate = result
            self.untried = list(reversed(moves))
            self.visits = 0
            self.value = 0.0

    root = Node(-1, color, (None, candidates, advantage, False))
    original = position.copy()
    gui_active = (
        pg.get_init() and pg.display.get_init() and pg.display.get_surface() is not None
    )
    next_event_check = started

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if gui_active and now >= next_event_check:
            pg.event.pump()
            if pg.event.peek(pg.QUIT):
                break
            next_event_check = now + 0.03
        position[:] = original
        node = root
        path = [root]
        while not node.untried and node.children:
            exploration = 2.0 * np.log(max(1, node.visits))
            maximizing = node.player == color

            def selection_score(child: Node) -> float:
                mean = child.value / child.visits
                exploitation = mean if maximizing else 1.0 - mean
                return float(exploitation + (exploration / child.visits) ** 0.5)

            node = max(node.children, key=selection_score)
            position[node.move] = -node.player
            path.append(node)

        if node.winner is None and node.untried:
            move = node.untried.pop()
            position[move] = node.player
            child = Node(move, -node.player, analyse(-node.player, 10))
            node.children.append(child)
            node = child
            path.append(node)

        winner = node.winner
        player = node.player
        evaluation = node.advantage
        rollout_moves = list(reversed(node.untried))
        immediate = node.immediate
        if winner is None and not rollout_moves:
            winner, rollout_moves, evaluation, immediate = analyse(player, 6)
        for _ in range(16):
            if winner is not None or time.perf_counter() >= deadline:
                break
            if immediate:
                winner = player
                break
            if not rollout_moves:
                break
            choice = 0
            if len(rollout_moves) > 1 and np.random.random() < 0.3:
                choice = int(np.random.randint(min(6, len(rollout_moves))))
            position[rollout_moves[choice]] = player
            player = -player
            winner, rollout_moves, evaluation, immediate = analyse(player, 6)

        if winner is None:
            reward = float(0.5 + 0.45 * np.tanh(color * evaluation / 400.0))
        else:
            reward = float(winner == color)
        for visited in path:
            visited.visits += 1
            visited.value += reward

    if root.children:
        chosen = max(
            root.children, key=lambda child: (child.visits, child.value / child.visits)
        )
        r, c = divmod(chosen.move, cols)
    else:
        r, c = divmod(candidates[0], cols)

    ###########################################################################
    return r, c


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
