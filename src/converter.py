import numpy as np
import cv2
import math
GCODE_PATH = "run.gcode"

# everything in milimeters
HEIGHT = 380
WIDTH = 345
FULL_PUZZLE_HEIGHT = 200
FULL_PUZZLE_WIDTH = 270
PUZZLE_HEIGHT = 200 / 5
PUZZLE_WIDTH = 270 / 6
WORK_AREA = np.array([
    [0, 0],         # bottom left
    [HEIGHT, 0],    # bottom right
    [0, WIDTH],     # top left
    [HEIGHT, WIDTH] # top right
], dtype=np.float32)

# in pixels
WORK_AREA_P = np.array([
    [514, 870],   # bottom left
    [1302, 874],  # bottom right
    [507, 138],   # top left
    [1315, 141]   # top right
], dtype=np.float32)

PERSPECTIVE = cv2.getPerspectiveTransform(WORK_AREA_P, WORK_AREA)

def get_coordinates_from_img(positions):
    for (x, y), z in positions:
        p = np.array([[[x, y]]], dtype=np.float32)
        mm = cv2.perspectiveTransform(p, PERSPECTIVE)
        x = mm[0, 0, 0]
        y = mm[0, 0, 1]
    return positions


def get_coordinates(positions):
    coordinates = []
    for row, col in positions:
        x = col * PUZZLE_WIDTH + PUZZLE_WIDTH / 2
        y = row * PUZZLE_HEIGHT + PUZZLE_HEIGHT / 2
        coordinates.append((x, y))
    return coordinates

# gcode
def lift(f):
    f.write("M3 S70\n")
    f.write("M5\n")

def lower(f):
    f.write("M3 S149\n")

def rotate(f, z_in):
    f.write(f"G0 Z{int(z_in)}\n")

def move_to(f, x, y):
    f.write(f"G1 X{int(x)} Y{int(y)}\n")

def convert_to_gcode(positions_in, positions_out):
    coords_in = get_coordinates_from_img(positions_in)
    coords_out = get_coordinates(positions_out)
    with open(GCODE_PATH, "w") as f:
        for ((x_in, y_in), z_in), (x_out, y_out) in zip(coords_in, coords_out):
            if ( not (0 <= x_in <= WIDTH and 0 <= y_in <= HEIGHT)):
                continue
            z_in = math.degrees(z_in)

            move_to(f, x_in, y_in)

            lower(f)
            lift(f)

            move_to(f, x_out, y_out)

            rotate(f, z_in)
            lower(f)
            lift(f)
            rotate(f, -z_in)