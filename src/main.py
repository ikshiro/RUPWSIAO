from jigsaw_solver import JigsawSolver
from multiprocessing import freeze_support


IMAGE_PATH = "zdjecia/kilka.jpg"

if __name__ == '__main__':
    # Required for Windows/frozen executables
    freeze_support()
    jigsaw_solver = JigsawSolver(IMAGE_PATH)
    jigsaw_solver.solve()


