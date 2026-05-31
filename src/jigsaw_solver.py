import math

import cv2

from detection import PuzzleDetector
from puzzle import Puzzle
from puzzle import EdgeType
import numpy as np
import copy
import converter



class JigsawSolver:

    def __init__(self, path):
        self.path = path
        self.puzzle_detector = PuzzleDetector(path)
        self.puzzles = []
        self.completed_puzzles = []


    def solve(self):
        self._get_puzzle_data()

        puzzles_number = len(self.puzzles)


                    
        directions = {
            "left": (-1,0),
            "right": (1,0),
            "up": (0,1),
            "down": (0,-1)
        }
        # col * puzzles_number + row
        puzzle_hashmap = {
            0: True
        }

        placed_puzzles = 1
        self.puzzles[0].is_placed = True

        while placed_puzzles != puzzles_number:
            for inspected_puzzle in self.puzzles:
                if not inspected_puzzle.is_placed:
                    continue
                for side in inspected_puzzle.puzzle_edges.keys():

                    if self.get_position_hash(
                        inspected_puzzle.position[0] + directions[side][0],
                        inspected_puzzle.position[1] + directions[side][1]) in puzzle_hashmap:
                        inspected_puzzle.puzzle_edges[side] = True
                        continue
                    
                    best_score = 0
                    best_index = None
                    for j, puzzle in enumerate(self.puzzles):
                        if puzzle.is_placed:
                            continue
                        
                        similarity = inspected_puzzle.compare(puzzle, side)

                        if similarity > best_score:
                            best_score = similarity
                            best_index = j
                    if best_index == None:
                        break

                    puzzle = self.puzzles[best_index]
                    puzzle.is_placed = True
                    position = (
                        inspected_puzzle.position[0] + directions[side][0],
                        inspected_puzzle.position[1] + directions[side][1]
                    )
                    puzzle.set_position(position)
                    puzzle_hashmap[self.get_position_hash(position[0], position[1])] = True
                    

                    puzzle_hashmap
                    placed_puzzles += 1
                
                if placed_puzzles == puzzles_number:
                    break


        positions_in = [(puzzle.center, puzzle.rotation) for puzzle in self.puzzles]
        positions_out = [puzzle.position for puzzle in self.puzzles]
        for piece in self.puzzles:
            print(piece.position)
            #piece.show()


        converter.convert_to_gcode(positions_in, positions_out)
                  


    def _get_puzzle_data(self):
        masks, boxes = self.puzzle_detector.predict()
        for i in range(len(masks)):
            puzzle = Puzzle(masks[i], boxes[i], self.path)
            self.puzzles.append(puzzle)
    

    def get_position_hash(self, col, row):
        return col*len(self.puzzles)+row