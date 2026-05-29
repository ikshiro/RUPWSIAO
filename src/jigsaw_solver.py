import math

from detection import PuzzleDetector
from puzzle import Puzzle
from puzzle import EdgeType
import numpy as np
import copy


IMAGE_PATH = "zdjecia/puzzle.jpg"


class JigsawSolver:
    puzzle_detector = PuzzleDetector()
    puzzles = []
    grid = []


    def solve(self):
        self._get_puzzle_data()

        puzzles_placed = 0
        puzzles_number = len(self.puzzles)

        corner_pairs = [
            ("down", "left")
            ("up", "left"),
            ("up", "right"),
            ("down", "right"),
        ]

        start_puzzle: Puzzle = None

        for puzzle in self.puzzles:
            i = 0
            print(puzzle.edges_types)
            for side1, side2 in corner_pairs:
                if (
                    puzzle.edges_types[side1] == EdgeType.FLAT and
                    puzzle.edges_types[side2] == EdgeType.FLAT
                ):
                    start_puzzle = copy.copy(puzzle)
                    break
                i += 1

                
            if start_puzzle is not None:
                puzzles_placed += 1
                start_puzzle.rotate(i*math.pi/2)
                break



    def _get_puzzle_data(self):
        masks, boxes = self.puzzle_detector.predict()
        for i in range(len(masks)):
            puzzle = Puzzle(masks[i], boxes[i])
            self.puzzles.append(puzzle)
    
