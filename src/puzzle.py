from enum import Enum
from PIL import Image
import numpy as np
import math


class EdgeType(Enum):
    FLAT = 0,
    FEMALE = 1,
    MALE = 2,
    UNDEFINED = 3


class Puzzle:
    data = []
    edges_types = dict(
        left = EdgeType.UNDEFINED,
        right = EdgeType.UNDEFINED,
        up = EdgeType.UNDEFINED,
        down = EdgeType.UNDEFINED)
    rotation = 0.0


    def __init__(self, contour):
        self._rotate_to_right_angle(contour)
        self._add_edges_types(contour)


    def _rotate_to_right_angle(self, contour):
        p1 = contour[0]
        p2 = contour[3]
        coeff = np.array([p1.x, 1], [p2.x, 1])
        const = np.array([p1.y, p2.y])
        solution = np.linalg.solve(coeff, const)
        x_distance = abs(p1.x - p2.x)
        y_distance = abs(p1.y - p2.y)
        self.rotation = math.tan(y_distance / x_distance)




    def _add_edges_types(self, contour):
        pass