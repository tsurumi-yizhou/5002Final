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
    Immutable geometry and pattern tables are built lazily by this entry point.
    Mutable search positions remain local; no GUI initialization is required.
    """
    ###########################################################################
    started = time.perf_counter()
    hard_end = started + 4.5
    board, edgeH, edgeV = state
    rows, cols = board.shape
    size = rows * cols
    # Offline training replaces only these two literals. Inference needs no files.
    _learned_policy_data = (
        (0, 8), (1, 86), (2, 73), (5, 48), (6, 35), (7, 5),
        (8, -8), (9, -35), (10, 64), (11, 26), (12, 11), (13, -7),
        (14, -31), (15, -211), (16, -196), (17, -211), (18, -120), (19, 7),
        (21, 61), (22, 50), (23, 16), (24, -26), (26, 12), (27, 19),
        (28, -10), (29, -7), (31, 7), (32, -19), (33, -19), (34, -15),
        (37, -6), (38, 2), (39, 8), (41, 48), (42, 26), (43, 4),
        (44, -21), (46, -7), (47, -13), (48, -10), (49, -12), (51, 89),
        (52, 40), (53, 1), (54, -10), (56, -9), (57, -4), (58, 1),
        (59, 8), (80, 54), (82, 73), (83, 55), (84, 1), (85, 12),
        (87, 12), (88, -4), (89, -6), (90, 15), (92, 1), (93, -7),
        (94, -1), (95, -11), (97, -1), (98, 16), (99, 7), (102, 19),
        (103, 15), (104, -7), (107, 13), (108, 20), (109, -5), (112, -1),
        (113, -1), (114, -8), (118, 1), (119, -1), (122, 6), (123, -13),
        (124, -15), (127, -4), (128, -8), (129, -11), (132, 10), (134, -12),
        (137, 1), (138, 1), (160, 26), (162, 35), (163, 17), (164, -12),
        (165, 13), (167, -12), (168, -8), (169, -11), (170, 73), (172, 55),
        (173, 6), (174, -5), (175, -8), (177, 4), (178, 7), (179, 9),
        (182, 4), (183, -5), (184, -13), (187, -8), (188, -9), (189, -8),
        (192, -11), (193, -2), (194, -9), (197, -1), (198, -1), (199, -1),
        (202, 108), (203, 39), (204, 17), (207, 2), (208, -3), (209, -11),
        (212, 90), (213, 42), (214, -7), (218, 1), (219, -1), (320, 41),
        (321, 71), (325, 7), (326, 2), (329, -2), (330, 15), (331, 4),
        (333, -3), (334, -4), (335, -12), (336, -5), (338, 27), (339, 10),
        (341, 12), (343, 19), (344, -2), (346, -1), (348, 14), (349, -1),
        (354, -4), (361, 13), (363, -5), (364, -5), (366, 2), (368, -2),
        (369, -2), (371, 17), (374, -3), (376, -1), (378, 2), (400, 32),
        (403, 31), (404, 17), (405, 2), (408, 3), (409, -1), (410, 5),
        (413, 2), (414, 1), (418, 1), (433, 34), (434, 34), (438, -2),
        (439, -1), (443, -3), (444, -4), (448, -2), (449, -1), (454, -4),
        (458, 7), (480, 1), (483, 5), (484, -14), (485, 2), (488, 1),
        (489, -5), (490, 13), (493, -3), (494, -2), (495, 4), (498, -2),
        (499, 1), (503, -2), (504, -10), (508, -4), (509, -1), (513, -2),
        (514, -3), (518, -1), (519, -1), (523, 4), (524, 4), (528, -2),
        (529, 6), (533, -1), (534, -1), (538, 1), (539, 1), (640, 5),
        (641, 21), (645, 6), (646, -3), (648, -5), (649, -3), (650, 19),
        (651, 20), (653, 12), (654, -3), (655, -15), (656, -6), (658, 13),
        (659, 17), (661, 12), (663, -6), (664, -2), (666, -1), (668, -1),
        (671, 3), (673, -1), (674, -3), (676, -1), (681, 40), (683, 35),
        (684, 8), (686, 2), (688, -2), (689, -2), (691, 53), (693, 46),
        (694, -3), (696, 3), (698, 3), (699, 1), (720, -2), (723, -1),
        (724, -7), (725, 9), (728, 4), (730, 1), (734, -3), (735, -1),
        (738, 1), (743, -6), (744, -5), (749, -4), (753, -6), (763, -7),
        (764, -1), (769, -5), (773, -2), (774, 1), (778, -3), (779, 2),
        (800, 86), (803, 99), (804, 45), (805, 2), (808, 6), (809, 6),
        (810, 75), (813, 77), (814, -3), (815, 5), (818, 7), (819, 7),
        (823, 3), (824, -6), (828, -4), (829, -3), (833, -3), (834, -3),
        (839, 4), (843, 204), (844, 21), (848, 13), (849, 40), (858, 1),
        (859, 8), (1285, -1), (1286, 1), (1290, 6), (1291, 4), (1294, -3),
        (1295, -19), (1296, -14), (1297, 23), (1299, 20), (1301, 4), (1302, 4),
        (1304, 2), (1306, 1), (1309, -1), (1311, 2), (1312, 1), (1314, -1),
        (1321, 3), (1322, 3), (1324, -4), (1326, -1), (1327, 1), (1329, 1),
        (1331, 16), (1332, -1), (1334, -2), (1337, 2), (1339, -1), (1360, 23),
        (1362, 7), (1365, 8), (1367, 3), (1369, -1), (1370, -1), (1372, -1),
        (1374, 4), (1375, -1), (1377, 1), (1392, -2), (1394, 23), (1397, -1),
        (1409, -2), (1412, 6), (1414, -1), (1417, -1), (1440, 2), (1442, 10),
        (1444, -5), (1445, 1), (1447, 1), (1449, 4), (1450, 5), (1452, 3),
        (1454, -1), (1455, -1), (1457, 1), (1459, -1), (1464, -1), (1467, 1),
        (1469, -2), (1472, -2), (1474, -1), (1477, 1), (1482, 11), (1484, 1),
        (1487, -3), (1489, 5), (1492, 10), (1494, -3), (1606, 4), (1610, 2),
        (1611, 1), (1615, -3), (1619, 4), (1631, 1), (1634, 4), (1636, 2),
        (1639, -2), (1641, 1), (1649, 3), (1651, 3), (1656, 1), (1724, 48),
        (1729, -2), (1734, 5), (1739, 4), (1760, -2), (1764, 2), (1765, 2),
        (1769, -2), (1784, -1), (1789, -2), (1804, 1), (1809, -3), (1814, -1),
        (1925, -1), (1926, -2), (1929, 1), (1931, -1), (1934, 2), (1936, 3),
        (1939, -1), (1941, 4), (1944, -1), (1946, -1), (1949, 3), (1951, -1),
        (1961, 9), (1964, -2), (1966, -1), (1969, -1), (1971, 6), (1974, -1),
        (1979, 1), (2009, 7), (2014, -5), (2019, -1), (2024, -3), (2029, 1),
        (2034, -3), (2039, -1), (2054, -2), (2059, -1), (2080, 5), (2084, 9),
        (2085, -2), (2089, -1), (2090, 6), (2094, -3), (2095, -1), (2099, 2),
        (2104, -2), (2109, -1), (2119, 1), (2124, 1), (2129, -1), (2134, -3),
        (2565, 1), (2566, 2), (2567, 4), (2569, -3), (2570, 1), (2571, 3),
        (2574, -1), (2575, -16), (2576, -6), (2577, 3), (2579, 10), (2581, 4),
        (2582, 1), (2584, -5), (2587, -1), (2589, -2), (2591, -2), (2592, -2),
        (2596, 1), (2599, 1), (2601, 20), (2602, 20), (2604, 17), (2606, -2),
        (2607, 1), (2609, -1), (2611, 5), (2612, 6), (2614, -1), (2645, -2),
        (2650, 3), (2654, -2), (2657, -1), (2662, 1), (2669, -2), (2672, -1),
        (2674, -2), (2677, -1), (2682, 2), (2684, -2), (2692, -2), (2694, -1),
        (2697, 2), (2720, 36), (2722, 23), (2725, 1), (2727, 2), (2729, 2),
        (2730, 3), (2732, 3), (2734, 1), (2735, 12), (2737, 10), (2739, 13),
        (2742, 3), (2747, -2), (2749, -2), (2752, -1), (2754, -1), (2759, -1),
        (2762, 83), (2764, 75), (2767, 7), (2769, 5), (2777, 2), (2779, 1),
        (2885, 5), (2890, -3), (2891, -3), (2894, -2), (2895, 2), (2896, -2),
        (2899, -1), (2901, 8), (2904, -1), (2906, 1), (2911, 2), (2914, 4),
        (2921, -4), (2926, -1), (2929, 1), (2931, 4), (2934, -2), (2965, -1),
        (2969, -1), (2970, -2), (2974, -2), (2975, 1), (2979, -1), (2984, -4),
        (2994, 5), (2999, 1), (3004, -2), (3009, -2), (3014, -3), (3019, -1),
        (3040, -6), (3045, -2), (3050, 1), (3054, -3), (3055, 2), (3059, -1),
        (3064, -1), (3079, -1), (3084, -3), (3089, 1), (3205, 4), (3206, 3),
        (3209, 8), (3211, 1), (3215, 6), (3216, 7), (3219, 15), (3221, 2),
        (3224, 4), (3229, 2), (3234, 1), (3239, -1), (3241, 94), (3244, 81),
        (3246, 5), (3249, 15), (3259, 1), (3285, -1), (3289, 6), (3290, -1),
        (3294, -1), (3295, -1), (3299, -1), (3304, -1), (3314, -1), (3319, -1),
        (3324, -3), (3329, -1), (3334, -1), (3360, 284), (3365, 21), (3369, 34),
        (3375, 37), (3379, 15), (3384, 26), (3389, 3), (3394, 3), (5127, 1),
        (5130, -1), (5131, -1), (5133, 3), (5135, -27), (5136, -15), (5137, 8),
        (5138, 22), (5141, 1), (5142, 1), (5147, 1), (5151, -1), (5152, 3),
        (5156, -1), (5161, 3), (5162, 2), (5163, -1), (5166, -1), (5168, 2),
        (5171, 3), (5172, 2), (5173, -1), (5178, -1), (5210, 1), (5215, -1),
        (5217, 1), (5232, 1), (5233, 2), (5242, 1), (5243, 2), (5252, 2),
        (5253, -1), (5257, 1), (5288, -1), (5290, 2), (5292, 1), (5295, 2),
        (5297, 1), (5298, 5), (5303, -1), (5307, -2), (5308, 1), (5312, -1),
        (5313, 1), (5322, 4), (5323, 2), (5327, 4), (5332, 11), (5333, -2),
        (5337, -1), (5456, 3), (5458, 5), (5461, 4), (5471, 4), (5473, 4),
        (5476, 1), (5481, -1), (5483, -1), (5498, 4), (5563, -3), (5568, 2),
        (5573, -5), (5578, -2), (5605, 3), (5608, 1), (5610, -1), (5628, 1),
        (5638, 2), (5643, 3), (5648, 1), (5653, 6), (5771, 2), (5773, 1),
        (5775, 3), (5776, 3), (5778, -4), (5781, 2), (5788, 2), (5793, -1),
        (5801, 2), (5803, 1), (5808, -1), (5811, 4), (5813, 4), (5845, -1),
        (5848, 1), (5850, 1), (5855, 3), (5858, -1), (5863, 2), (5868, 1),
        (5873, -2), (5883, -2), (5888, -1), (5893, 1), (5925, -1), (5928, 1),
        (5930, 9), (5933, 13), (5935, -1), (5943, -2), (5948, -3), (5963, 10),
        (5968, -1), (6406, -1), (6415, -5), (6416, 2), (6417, 18), (6432, 5),
        (6436, 2), (6441, -1), (6446, 1), (6451, 1), (6452, 1), (6532, 1),
        (6565, 2), (6577, 1), (6592, -1), (6602, 1), (6607, -1), (6612, 6),
        (6766, -1), (6776, -1), (6885, -2), (6890, -1), (6895, 1), (7051, -1),
        (7056, -1), (7066, -1), (7081, -1), (7091, 1), (7130, -2), (7135, 2),
        (7205, -1), (7210, 1), (7215, -1), (7685, 1), (7686, -1), (7687, -1),
        (7695, -4), (7696, -4), (7706, -1), (7712, -1), (7717, -1), (7721, -1),
        (7722, -1), (7727, -1), (7732, 1), (7736, 1), (7737, 1), (7765, -1),
        (7772, 3), (7775, -1), (7777, 2), (7787, -1), (7807, -1), (7845, -1),
        (7850, 4), (7852, 2), (7857, 3), (7862, -1), (7872, -1), (7877, 1),
        (7882, 7), (7887, 1), (8015, -2), (8016, 2), (8036, -2), (8041, 1),
        (8046, -2), (8056, -1), (8085, -2), (8090, -1), (8165, -1), (8170, -1),
        (8325, -1), (8326, 2), (8330, 7), (8331, 5), (8335, -1), (8351, -1),
        (8356, 1), (8361, 13), (8366, -1), (8415, -1), (8485, -1), (8495, -1),
        (10245, 3), (10246, 1), (10248, -1), (10250, -1), (10251, -1), (10252, -1),
        (10255, -29), (10256, -23), (10257, -2), (10258, 35), (10266, 1), (10267, 1),
        (10271, -1), (10272, -1), (10276, -1), (10277, 3), (10278, 1), (10282, 1),
        (10286, -1), (10287, -1), (10288, -1), (10291, 1), (10292, 3), (10293, 1),
        (10330, -1), (10332, 1), (10347, -1), (10348, 1), (10353, -1), (10362, -2),
        (10363, -3), (10367, -1), (10372, -2), (10373, -1), (10377, 3), (10405, 1),
        (10410, 1), (10415, 6), (10417, 6), (10418, 7), (10427, -1), (10433, 1),
        (10437, -1), (10438, 9), (10442, 2), (10443, 1), (10447, 5), (10448, 8),
        (10457, 1), (10458, -1), (10566, 1), (10568, 1), (10573, -1), (10575, -4),
        (10576, 2), (10578, 2), (10586, -2), (10591, -1), (10596, -1), (10601, -1),
        (10606, 2), (10608, -1), (10613, 1), (10650, 1), (10653, 1), (10655, 1),
        (10673, 1), (10678, -1), (10683, -1), (10688, -2), (10698, -1), (10725, -1),
        (10728, 1), (10733, 1), (10738, 2), (10748, -2), (10753, -1), (10758, 1),
        (10763, -1), (10768, -1), (10778, -1), (10888, -1), (10891, 1), (10895, 16),
        (10896, 16), (10898, 21), (10921, 9), (10926, 7), (10928, 9), (10931, -1),
        (10936, 3), (10938, 11), (10968, -1), (10970, -1), (10975, 3), (10978, 2),
        (10988, -1), (11008, -1), (11018, 2), (11045, 13), (11048, 7), (11055, 26),
        (11058, 22), (11068, 4), (11073, 6), (11525, -1), (11526, -1), (11527, -1),
        (11531, -1), (11532, -1), (11535, -7), (11536, -5), (11537, -4), (11546, 5),
        (11551, -1), (11556, -1), (11566, -1), (11567, -1), (11571, -1), (11605, 1),
        (11607, 8), (11612, -1), (11615, 1), (11617, 2), (11632, -1), (11642, -1),
        (11652, 3), (11657, 1), (11685, -1), (11690, 1), (11695, -1), (11697, -3),
        (11712, -1), (11722, 2), (11727, -1), (11732, 1), (11737, 1), (11845, 2),
        (11851, 1), (11855, 1), (11856, 2), (11871, -2), (11930, 3), (11935, 2),
        (12005, -1), (12015, -4), (12165, -1), (12166, -1), (12171, -1), (12176, -2),
        (12186, -1), (12191, -1), (12206, -1), (12211, 1), (12216, 2), (12245, 1),
        (12250, -1), (12255, -1), (12325, -2), (12330, 1), (12335, -1), (12805, -1),
        (12806, -1), (12807, 1), (12812, 2), (12815, 2), (12816, 17), (12817, 78),
        (12826, 1), (12827, 1), (12836, 1), (12846, 1), (12856, -2), (12885, -1),
        (12890, 1), (12892, 1), (12897, 4), (12912, -1), (12927, 1), (12937, 1),
        (12965, 9), (12967, 5), (12975, 56), (12977, 36), (12987, 1), (12992, 2),
        (12997, 4), (13125, -1), (13126, 1), (13130, 1), (13131, -1), (13136, 3),
        (13146, -1), (13156, 1), (13166, -1), (13205, -2), (13210, 2), (13215, 3),
        (13285, -1), (13445, 2), (13446, 13), (13451, 2), (13455, 53), (13456, 113),
        (13466, 8), (13471, 2), (13476, 1), (13525, 12), (13530, 1), (13535, 4),
        (20485, 6), (20486, 1), (20487, 1), (20490, -1), (20492, -1), (20493, -1),
        (20495, -23), (20496, -32), (20497, -17), (20498, 10), (20499, 25), (20511, 4),
        (20512, -1), (20513, -1), (20514, -1), (20516, -1), (20518, 1), (20519, 3),
        (20539, 1), (20568, 1), (20569, 1), (20575, 2), (20577, -1), (20578, 4),
        (20593, 1), (20612, 1), (20613, 2), (20614, -1), (20648, 1), (20650, 3),
        (20652, 1), (20655, -1), (20658, 1), (20659, -1), (20692, 4), (20810, 1),
        (20815, 3), (20816, -1), (20818, 2), (20834, -1), (20893, 1), (20894, 2),
        (20928, 2), (20969, 1), (20975, -1), (20978, 1), (21008, 1), (21013, 2),
        (21014, 1), (21018, 1), (21019, 1), (21131, 1), (21135, 2), (21136, -1),
        (21138, 2), (21139, -3), (21146, -1), (21151, 1), (21171, 2), (21173, 1),
        (21205, -1), (21208, 1), (21209, -1), (21210, -2), (21215, -1), (21218, -1),
        (21219, -1), (21228, -1), (21229, 1), (21233, -1), (21238, -1), (21253, -1),
        (21254, 1), (21290, 9), (21293, 2), (21313, 2), (21328, -2), (21329, 2),
        (21775, -2), (21776, -4), (21777, 19), (21779, -1), (21816, 1), (21852, 4),
        (21857, 1), (21892, 3), (21894, 11), (21929, -1), (21939, -1), (21947, 1),
        (22096, -1), (22129, -1), (22254, -1), (22259, -3), (22269, 5), (22274, -1),
        (22294, -3), (22410, -1), (22415, -1), (22416, -1), (22419, 3), (22459, -1),
        (22485, -1), (22499, -1), (22509, 2), (22569, -1), (22574, 3), (22575, 2),
        (22579, 1), (22599, 1), (22609, -1), (22614, 2), (23055, -4), (23057, 4),
        (23069, 1), (23135, -1), (23137, 2), (23139, 1), (23152, -1), (23154, -1),
        (23227, 1), (23232, -1), (23249, -2), (23370, 1), (23375, 1), (23376, 1),
        (23386, -1), (23411, 1), (23450, 1), (23469, -1), (23474, 1), (23494, -1),
        (23535, 3), (23554, -1), (23574, -1), (23579, 1), (23690, -1), (23694, -1),
        (23695, 1), (23714, 2), (23726, 3), (23729, 3), (23736, 3), (23739, 2),
        (23779, 1), (23789, -2), (23794, -2), (23845, -1), (23855, 1), (23859, 1),
        (23869, -1), (23874, 2), (23879, -1), (25612, 1), (25615, -6), (25616, -11),
        (25617, 16), (25618, 21), (25632, -1), (25633, -1), (25692, 1), (25697, 1),
        (25698, -1), (25732, 1), (25733, 1), (25737, -1), (25775, -1), (25778, -1),
        (25788, -1), (25807, -1), (25808, -1), (25812, 2), (25935, -2), (25936, 9),
        (25938, -3), (25976, -1), (25978, -2), (26093, -1), (26095, -1), (26098, -1),
        (26128, -1), (26133, 3), (26256, -4), (26258, 1), (26276, -1), (26298, -2),
        (26348, -1), (26433, -1), (26448, -1), (26892, 1), (26895, -2), (26896, 13),
        (26897, 18), (26926, -1), (26927, -1), (26936, -1), (26937, -1), (27052, -1),
        (27055, 1), (27057, 2), (27087, -1), (27092, 1), (27375, -2), (27536, 2),
        (27551, -1), (27556, -1), (27566, -1), (27576, -1), (27690, 4), (27695, 1),
        (28170, -1), (28171, 1), (28172, -1), (28175, -9), (28177, -2), (28196, 1),
        (28197, -1), (28217, -1), (28250, -1), (28255, -2), (28257, 1), (28277, -1),
        (28292, -1), (28330, 3), (28332, 1), (28335, 4), (28337, -2), (28490, -1),
        (28495, 2), (28496, -2), (28536, -1), (28570, -1), (28575, -2), (28810, -1),
        (28811, 1), (28815, -1), (28816, -1), (28836, -1), (28846, 2), (28895, -3),
        (30735, -21), (30736, -21), (30737, -4), (30738, -8), (30776, -1), (30777, -1),
        (30778, -1), (30815, -1), (30817, -1), (30818, -2), (30895, -1), (30897, 8),
        (30898, -1), (30932, 1), (31051, -1), (31055, -2), (31056, -2), (31058, -2),
        (31071, -1), (31093, 1), (31096, 1), (31153, -1), (31218, -1), (31375, -4),
        (31376, -2), (31378, -2), (31418, 1), (31455, -1), (31458, -1), (31533, 1),
        (31535, 3), (31538, 3), (32012, -1), (32015, -2), (32016, -6), (32017, -3),
        (32037, 1), (32056, -1), (32097, 4), (32112, -1), (32175, 1), (32177, -1),
        (32207, -1), (32217, -1), (32330, -1), (32335, 2), (32336, -2), (32351, 1),
        (32376, -1), (32415, -1), (32495, 5), (32655, 1), (32656, -1), (32810, 5),
        (33290, -1), (33292, -1), (33295, -9), (33296, -6), (33297, -4), (33316, -1),
        (33331, 1), (33336, -2), (33337, 2), (33375, 1), (33377, -1), (33455, 15),
        (33457, 16), (33611, -1), (33615, 3), (33616, 1), (33690, -1), (33695, -1),
        (33775, 10), (33935, 11), (33936, 20), (33956, 2), (40970, 1), (40971, -1),
        (40975, -21), (40976, -26), (40977, -14), (40978, 24), (40979, 57), (40996, 1),
        (41013, -1), (41055, -1), (41093, 1), (41172, 2), (41173, 3), (41174, 1),
        (41291, -1), (41295, 1), (41296, -2), (41298, 2), (41299, 3), (41319, 1),
        (41393, -4), (41394, -1), (41413, 2), (41454, -2), (41455, 2), (41458, 1),
        (41459, -1), (41493, -1), (41494, -1), (41498, -1), (41616, 3), (41618, 6),
        (41619, 3), (41651, 3), (41653, 1), (41693, 1), (41698, 1), (41713, -1),
        (41770, 4), (41773, 1), (41775, 2), (41794, 2), (42255, -8), (42256, -3),
        (42257, -5), (42259, 16), (42297, -1), (42335, -1), (42352, -2), (42354, -1),
        (42374, 1), (42415, -1), (42417, -1), (42419, -1), (42434, 1), (42452, 1),
        (42454, 1), (42459, 1), (42575, 1), (42576, 1), (42579, -1), (42591, -1),
        (42596, -1), (42599, -1), (42611, 1), (42614, 1), (42650, -1), (42655, 1),
        (42659, -1), (42694, -1), (42699, -1), (42730, -1), (42739, 1), (42774, -1),
        (42895, -1), (42896, -2), (42899, -1), (42911, -1), (42914, 1), (42934, 1),
        (42975, 2), (42994, -1), (43050, 2), (43054, -1), (43055, -2), (43074, -1),
        (43094, -1), (43535, -2), (43536, -3), (43537, 25), (43539, 22), (43577, 1),
        (43617, 1), (43637, -1), (43652, -1), (43654, 1), (43690, 10), (43692, 7),
        (43697, 12), (43699, 5), (43712, 2), (43717, 2), (43719, 2), (43855, -3),
        (43856, 1), (43939, -1), (43954, -1), (43974, 2), (44010, 2), (44019, -1),
        (44175, -1), (44176, 17), (44179, 12), (44191, 4), (44194, 3), (44255, 1),
        (44259, 3), (44274, 1), (44299, 2), (46095, -18), (46096, -19), (46097, -1),
        (46098, 2), (46116, -1), (46117, -1), (46118, 1), (46136, -1), (46137, 1),
        (46138, -1), (46178, 1), (46193, 1), (46257, -2), (46292, -1), (46415, -3),
        (46416, -3), (46418, 6), (46431, 6), (46436, 2), (46451, 1), (46456, -1),
        (46495, -3), (46533, -1), (46613, 3), (46735, -3), (46736, -1), (46738, -2),
        (46771, 1), (46773, 1), (46815, -1), (46818, -1), (46853, -1), (46895, -1),
        (46898, 1), (46913, -2), (47375, -12), (47376, -9), (47377, 2), (47396, -2),
        (47417, 2), (47455, -16), (47457, -3), (47492, 1), (47497, -1), (47535, 1),
        (47537, -2), (47557, -1), (47577, 1), (47695, 19), (47696, 25), (47736, 1),
        (47855, 1), (48015, 2), (48016, -1), (48056, 2), (48095, 2), (48656, -1),
        (48657, -1), (48697, -1), (48735, 1), (48815, -2), (48975, 1), (48976, -1),
        (48996, 1), (49055, -2), (49135, -1), (49295, -1), (49296, -4), (49316, -1),
        (49336, 1), (49375, 1), (49455, 4), (51215, -4), (51216, -2), (51217, 40),
        (51218, 110), (51238, 1), (51251, -1), (51258, 2), (51298, 2), (51318, 1),
        (51377, 1), (51378, 3), (51412, 1), (51535, -2), (51536, 2), (51538, 1),
        (51571, 1), (51615, 1), (51618, 1), (51718, -1), (51855, 2), (51856, 33),
        (51858, 34), (51938, 1), (51978, 1), (52495, -7), (52496, 8), (52497, -8),
        (52516, 1), (52517, 1), (52537, -1), (52575, -1), (52577, 1), (52655, 1),
        (52657, 1), (52815, 11), (52816, -5), (52975, -1), (53135, -3), (53136, 1),
        (53775, -4), (53776, 23), (53777, 124), (53796, 4), (53797, -1), (53855, 3),
        (53857, -2), (54095, 2), (54096, -3), (54175, 1), (54255, -2), (54495, -2),
        (61455, -122), (61456, -108), (61457, -94), (61476, 4), (61477, -1), (61478, -6),
        (61479, -4), (61496, -5), (61497, -3), (61499, -2), (61535, -9), (61537, -7),
        (61538, -4), (61539, -4), (61557, 2), (61558, -1), (61559, -2), (61577, -2),
        (61578, -1), (61579, -2), (61615, -6), (61617, -5), (61618, 7), (61619, -4),
        (61637, 1), (61638, -2), (61639, 1), (61657, 3), (61659, -1), (61775, -2),
        (61776, -3), (61796, 2), (61798, 1), (61816, 1), (61818, -1), (61858, 2),
        (61859, -1), (61878, -2), (61879, -4), (61898, -2), (61899, 1), (61938, 3),
        (61939, -1), (61978, 2), (61979, 3), (62095, -1), (62096, -7), (62118, -2),
        (62136, 1), (62138, 1), (62139, 3), (62178, -1), (62199, -1), (62218, -1),
        (62259, -1), (62278, -2), (62279, 1), (62298, 5), (62299, 3), (62756, -1),
        (62777, -1), (62779, 1), (62859, -1), (62897, -2), (62899, -1), (63076, 1),
        (63135, -1), (63416, 1), (63455, 1), (63479, 1), (63539, -1), (64037, 1),
        (64056, 2), (64057, 1), (64059, 5), (64119, 1), (64175, -1), (64199, -1),
        (64217, 1), (64219, 1), (64356, -1), (64376, -1), (64379, 1), (64459, 1),
        (64519, 2), (64696, 4), (64699, -1), (64815, -1), (66617, -1), (66916, 1),
        (67236, 1), (67256, -1), (67896, -1), (69257, -1), (69337, -1), (71736, -1),
        (71897, 1), (71898, 2), (72376, -5), (73177, 1),
    )
    _learned_policy_scale = 1.0
    training_recorder = getattr(best_move, "_training_recorder", None)
    policy_feature_map: dict[int, list[int]] = {}
    base_logits: dict[int, float] = {}

    def finish(move: int, reason: str, root=None, proof: int = 0) -> Point:
        point = divmod(int(move), cols)
        if training_recorder is not None:
            children = [] if root is None else [
                {"move": child.move, "visits": child.visits,
                 "value": child.value / child.visits if child.visits else 0.5,
                 "solved": child.solved}
                for child in root.children
            ]
            training_recorder({
                "move": list(point), "reason": reason, "color": color,
                "iterations": 0 if root is None else root.visits,
                "solved": proof if root is None else root.solved,
                "children": children, "policy_features": policy_feature_map,
                "base_logits": base_logits,
            })
        return point

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

    # Cache only the most recent board's geometry; stone positions are never cached.
    topology_key = (rows, cols, edgeH.tobytes(), edgeV.tobytes())
    topology = getattr(best_move, "_topology", None)
    if topology is None or topology[0] != topology_key:
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
        topology = (topology_key, windows, before, after, interiors, affected, centrality)
        best_move._topology = topology
    else:
        _, windows, before, after, interiors, affected, centrality = topology
    if not windows:
        return finish(max(empty, key=lambda point: (centrality[point], -point)),
                      "no_windows", proof=-1)

    count = [
        [sum(position[point] == player for point in points) for points in windows]
        for player in (+1, -1)
    ]
    empty_sum = [sum(point for point in points if position[point] == 0)
                 for points in windows]
    # 6 black counts x 6 white counts x 3 left x 3 right boundary states.
    # Intern equal entries so unchanged tactical/value states need no updates.
    table = getattr(best_move, "_window_table", None)
    if table is None:
        weights = (1, 6, 40, 320, 6000, 0)
        entries: list[tuple[int, int, int, int]] = []
        interned: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
        for black in range(6):
            for white in range(6):
                for left in (-1, 0, +1):
                    for right in (-1, 0, +1):
                        active_black = white == 0 and left != +1 and right != +1
                        active_white = black == 0 and left != -1 and right != -1
                        entry = (
                            weights[black] if active_black else 0,
                            weights[white] if active_white else 0,
                            black - 2 if active_black and black >= 3 else 0,
                            white - 2 if active_white and white >= 3 else 0,
                        )
                        entries.append(interned.setdefault(entry, entry))
        table = tuple(entries)
        best_move._window_table = table
    window_entries = [(0, 0, 0, 0)] * len(windows)
    totals = [0, 0]
    # Exact integer priorities avoid accumulating floating-point score drift.
    # 100 * own + 112 * opponent reproduces the existing 1.12 defence weight.
    priorities = [[0] * size for _ in range(2)]
    centre_scores = [100.0 * value for value in centrality]
    winning_count = [[0] * size for _ in range(2)]
    winning_points: list[set[int]] = [set(), set()]
    winning_window = [[-1] * len(windows) for _ in range(2)]
    threes: list[set[int]] = [set(), set()]
    fives = [set(), set()]

    def refresh(wid: int) -> None:
        entry = table[count[0][wid] * 54 + count[1][wid] * 9
                      + position[before[wid]] * 3 + position[after[wid]] + 4]
        old = window_entries[wid]
        if entry is old:
            return
        window_entries[wid] = entry
        black_difference, white_difference = entry[0] - old[0], entry[1] - old[1]
        if black_difference or white_difference:
            totals[0] += black_difference
            totals[1] += white_difference
            black_delta = 100 * black_difference + 112 * white_difference
            white_delta = 100 * white_difference + 112 * black_difference
            black_priority, white_priority = priorities
            for point in windows[wid]:
                black_priority[point] += black_delta
                white_priority[point] += white_delta
        for side in (0, 1):
            previous_type, new_type = old[side + 2], entry[side + 2]
            if previous_type == new_type:
                continue
            if previous_type == 1:
                threes[side].discard(wid)
            elif previous_type == 2:
                point = winning_window[side][wid]
                winning_count[side][point] -= 1
                if winning_count[side][point] == 0:
                    winning_points[side].discard(point)
                winning_window[side][wid] = -1
            elif previous_type == 3:
                fives[side].discard(wid)
            if new_type == 1:
                threes[side].add(wid)
            elif new_type == 2:
                point = empty_sum[wid]
                winning_count[side][point] += 1
                winning_points[side].add(point)
                winning_window[side][wid] = point
            elif new_type == 3:
                fives[side].add(wid)

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
        return priorities[side][point] + centre_scores[point]

    def ranked(player: int) -> list[int]:
        side = 0 if player == +1 else 1
        if winning_points[side]:
            choices = winning_points[side]
        elif len(winning_points[1 - side]) == 1:
            choices = winning_points[1 - side]
        else:
            choices = forks(player) or empty
        side_priority = priorities[side]
        return sorted(choices, key=lambda point: (
            -side_priority[point] - centre_scores[point], point
        ))

    def policy_features(player: int, moves: list[int]) -> dict[int, list[int]]:
        """Colour-relative seven-cell shapes, mirrored with the candidate slot.

        Digits: empty=0, own=1, opponent=2, disconnected boundary=3.
        A seven-digit base-four pattern and one of five slots need 81,920
        entries; symmetric directions share keys across randomly generated boards.
        """
        features: dict[int, list[int]] = {move: [] for move in moves}
        for wid, points in enumerate(windows):
            cells = (before[wid],) + points + (after[wid],)
            digits = [3 if point == size else
                      0 if position[point] == 0 else
                      1 if position[point] == player else 2 for point in cells]
            forward = sum(digit << (2 * index) for index, digit in enumerate(digits))
            backward = sum(digit << (2 * index)
                           for index, digit in enumerate(reversed(digits)))
            for slot, point in enumerate(points):
                if point in features:
                    features[point].append(min(forward * 5 + slot,
                                               backward * 5 + 4 - slot))
        return features

    root_side = 0 if color == +1 else 1
    candidates = ranked(color)
    if winning_points[root_side]:
        return finish(candidates[0], "immediate", proof=color)
    if len(candidates) == 1:
        return finish(candidates[0], "forced")
    proven = outcome(color)
    if proven:
        return finish(candidates[0], "proven", proof=proven)
    if len(empty) == size:
        return finish(candidates[0], "opening")
    if len(empty) == size - 1:
        first = next(point for point in range(size) if position[point])
        neighbours = [point for point in candidates
                      if max(abs(point // cols - first // cols),
                             abs(point % cols - first % cols)) <= 1]
        if neighbours:
            return finish(neighbours[0], "opening")

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
        return finish(line[0], "forcing_win", proof=color)
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
            return finish(ranked(color)[0], "forced_loss", proof=-color)
        if len(candidates) == 1:
            return finish(candidates[0], "forced")
        candidates.sort(key=lambda point: (point not in screened_moves,
                                           -move_score(point, color), point))

    policy_data = getattr(best_move, "_policy_data_override", _learned_policy_data)
    policy_scale = getattr(best_move, "_policy_scale_override", _learned_policy_scale)
    use_policy = bool(policy_data) and policy_scale > 0
    if use_policy or training_recorder is not None:
        policy_feature_map = policy_features(color, candidates)
        base_logits = {move: float(np.log1p(move_score(move, color) / 100.0))
                       for move in candidates}
    if use_policy:
        policy_cache = getattr(best_move, "_policy_cache", None)
        if policy_cache is None or policy_cache[0] is not policy_data:
            # Sparse source literals become a compact, constant-time dense lookup.
            policy_table = np.zeros(81920, dtype=np.int16)
            for key, weight in policy_data:
                policy_table[key] = weight
            best_move._policy_cache = (policy_data, policy_table)
        else:
            policy_table = policy_cache[1]
        learned_logits = {}
        for move, keys in policy_feature_map.items():
            bonus = (float(policy_table[keys].sum()) / (256.0 * len(keys) ** 0.5)
                     if keys else 0.0)
            learned_logits[move] = base_logits[move] + policy_scale * max(-0.8, min(0.8, bonus))
        candidates.sort(key=lambda move: (
            danger == 1 and move not in screened_moves, -learned_logits[move], move
        ))

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
                # Expand another legal move; skip proven losses.
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
                side_priority = priorities[side]
                move = max(empty, key=lambda point: (
                    side_priority[point] + centre_scores[point],
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
            return finish(chosen.move, "search", root)
        # Unexpanded legal moves are preferable to replies already proven to lose.
        if len(root.children) < len(root.moves):
            return finish(root.moves[len(root.children)], "search", root)
    return finish(candidates[0], "search", root)
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
