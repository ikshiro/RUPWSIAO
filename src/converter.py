GCODE_PATH = "run.gcode"
# everything in milimeters
HEIGHT = 300
WIDTH = 300
PUZZLE = 40



def get_coordinates_from_img():
    pass


def get_coordinates(positions):
    coordinates = []
    for row, col in positions:
        x = col * PUZZLE
        y = row * PUZZLE
        coordinates.append((x, y))


def convert_to_gcode(positions_in, positions_out):
    coords_in = get_coordinates(positions_in)
    coords_out = get_coordinates(positions_out)
    with open(GCODE_PATH, "w") as f:
        for x, y in coords_in:
            f.write(f"G1 X{x} Y{y} ;\n")