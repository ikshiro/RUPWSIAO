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

# grid corners:
# 514, 870  - bottom left
# 1302, 874 - bottom right
# 507, 138  - top left
# 1315, 141 - top right

# in pixels
WIDTH_PX  = ((1302-514) + (1315-507)) / 2
HEIGHT_PX = ((870-138) + (874-141)) / 2

SX = WIDTH / WIDTH_PX
SY = HEIGHT / HEIGHT_PX


def get_coordinates_from_img(positions):
    for (x, y), z in positions:
        x = (x - 507) * SX
        y = (y - 138) * SY
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