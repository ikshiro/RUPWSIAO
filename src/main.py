from jigsaw_solver import JigsawSolver
from multiprocessing import freeze_support


if __name__ == '__main__':
    # Required for Windows/frozen executables
    freeze_support()
    jigsaw_solver = JigsawSolver()
    jigsaw_solver.get_puzzle_data()


