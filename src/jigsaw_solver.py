import math

from detection import PuzzleDetector
from puzzle import Puzzle
from puzzle import EdgeType
import numpy as np
import copy



class JigsawSolver:

    def __init__(self, path):
        self.path = path
        self.puzzle_detector = PuzzleDetector(path)
        self.puzzles = []
        self.completed_puzzles = []


    def solve(self):
        self._get_puzzle_data()

        puzzles_number = len(self.puzzles)

        corner_pairs = [
            ("down", "left"),
            ("up", "left"),
            ("up", "right"),
            ("down", "right")
        ]

        inspected_puzzle: Puzzle = None

        for j in range(len(self.puzzles)):
            puzzle = self.puzzles[j]
            i = 0
            for side1, side2 in corner_pairs:
                if (
                    puzzle.edges_types[side1] == EdgeType.FLAT and
                    puzzle.edges_types[side2] == EdgeType.FLAT
                ):
                    inspected_puzzle = self.puzzles.pop(j)
                    inspected_puzzle.set_position((0, 0))
                    self.completed_puzzles.append(copy.copy(inspected_puzzle))
                    break
                i += 1

                
            if inspected_puzzle is not None:
                inspected_puzzle.rotate(i*math.pi/2)
                break

        directions = {
            "left": (-1,0),
            "right": (1,0),
            "up": (0,1),
            "down": (0,-1)
        }
        side = "right"

        while len(self.completed_puzzles) != puzzles_number:
            best_score = 0.0
            for j in range(len(self.puzzles)):
                puzzle = self.puzzles[j]
                similarity = inspected_puzzle.compare(puzzle, side)
                if similarity >= best_score:
                    best_score = similarity

                if best_score > 0.7:
                    puzzle.set_position(inspected_puzzle.position + directions[side])
                    inspected_puzzle = self.puzzles.pop(j)
                    self.completed_puzzles.append(copy.copy(inspected_puzzle)) 
                    break
            

            if best_score <= 0.7:
                if side == "up":
                    break
                else:
                    side = "up"
                    inspected_puzzle = copy.copy(self.completed_puzzles[-inspected_puzzle.position[0]-1])
                  


    def _get_puzzle_data(self):
        masks, boxes = self.puzzle_detector.predict()
        for i in range(len(masks)):
            puzzle = Puzzle(masks[i], boxes[i], self.path)
            self.puzzles.append(puzzle)
    
