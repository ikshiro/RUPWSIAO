import numpy as np
import cv2
GCODE_PATH = "run.gcode"

# everything in milimeters
HEIGHT = 345
WIDTH = 380
FULL_PUZZLE_HEIGHT = 200
FULL_PUZZLE_WIDTH = 270
PUZZLE_HEIGHT = 200 / 5
PUZZLE_WIDTH = 270 / 6
WORK_AREA = np.array([
    [0, 0],
    [HEIGHT, 0],
    [0, WIDTH],
    [HEIGHT, WIDTH]
], dtype=np.float32)

# in pixels
WORK_AREA_P = np.array([
    [514, 870],   # dl
    [1302, 874],  # rl
    [507, 138],   # tl
    [1315, 141]   # tr
], dtype=np.float32)

PERSPECTIVE = cv2.getPerspectiveTransform(WORK_AREA_P, WORK_AREA)


def get_coordinates_from_img(positions):
    for x, y in positions:
        p = np.array([[[x, y]]], dtype=np.float32)
        mm = cv2.perspectiveTransform(p, PERSPECTIVE)
        x = mm[0, 0, 0]
        y = mm[0, 0, 1]


def get_coordinates(positions):
    coordinates = []
    for row, col in positions:
        x = col * PUZZLE_WIDTH + PUZZLE_WIDTH / 2
        y = row * PUZZLE_HEIGHT + PUZZLE_HEIGHT / 2
        coordinates.append((x, y))


# gcode
def lift(f):
    f.write("M3 S55\n")
    f.write("M5\n")

def lower(f):
    f.write("M3 S149\n")

def vacuum_on(f):
    f.write("M9\n")

def vacuum_off(f):
    f.write("M8\n")

def move_to(f, x, y):
    f.write(f"G1 X{x} Y{y}\n")

def convert_to_gcode(positions_in, positions_out):
    coords_in = get_coordinates(positions_in)
    coords_out = get_coordinates(positions_out)
    with open(GCODE_PATH, "w") as f:
        for (x_in, y_in), (x_out, y_out) in zip(coords_in, coords_out):
            if ( not (0 <= x_in <= WIDTH and 0 <= y_in <= HEIGHT)):
                continue

            move_to(f, x_in, y_in)

            lower(f)
            vacuum_on(f)
            lift(f)

            move_to(f, x_out, y_out)

            lower(f)
            vacuum_off(f)
            lift(f)