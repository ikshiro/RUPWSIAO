from detection import PuzzleDetector


IMAGE_PATH = "zdjecia/puzzle.jpg"


class JigsawSolver:
    puzzle_detector = PuzzleDetector()
    puzzles = []

    def get_puzzle_data(self):
        self.puzzle_detector.predict()