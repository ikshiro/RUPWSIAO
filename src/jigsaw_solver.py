from detection import PuzzleDetector
from puzzle import Puzzle


IMAGE_PATH = "zdjecia/puzzle.jpg"


class JigsawSolver:
    puzzle_detector = PuzzleDetector()
    puzzles = []


    def solve(self):
        self._get_puzzle_data()


    def _get_puzzle_data(self):
        contours = self.puzzle_detector.predict()
        print(contours)
        for contour in contours:
            puzzle = Puzzle(contour)
            self.puzzles.append(puzzle)