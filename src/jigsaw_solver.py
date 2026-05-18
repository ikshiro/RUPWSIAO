from detection import PuzzleDetector
from puzzle import Puzzle
import numpy as np


IMAGE_PATH = "zdjecia/kilka.jpg"


class JigsawSolver:
    puzzle_detector = PuzzleDetector()
    puzzles = []


    def solve(self):
        self._get_puzzle_data()


    def _get_puzzle_data(self):
        masks, boxes = self.puzzle_detector.predict()
        for i in range(len(masks)):
            puzzle = Puzzle(masks[i], boxes[i], False)
            self.puzzles.append(puzzle)
