import detection
from multiprocessing import freeze_support


if __name__ == '__main__':
    # Required for Windows/frozen executables
    freeze_support()
    puzzle_detector = detection.PuzzleDetector()
    puzzle_detector.register_database()
    puzzle_detector.create_config()
    #puzzle_detector.train()
    puzzle_detector.predict()


