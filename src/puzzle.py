from enum import Enum
import numpy as np
import math
import cv2
from numpy.linalg import LinAlgError
from scipy import ndimage


IMAGE_PATH = "zdjecia/puzzle.jpg"


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
    box = []
    debug = False


    def __init__(self, contour, box, debug):
        self.debug = debug
        self.box = box.tensor.numpy()[0]
        self._rotate_to_right_angle(contour)
        self._add_edges_types(contour)


    def _rotate_to_right_angle(self, contour):
        i = 0
        frame_size = 8
        step = 3
        best_solution = [contour[0][0], contour[1][0]]
        best_hits = 0
        while True:
            if i*frame_size+frame_size*2 >= len(contour):
                break
            for offset in range(0, frame_size, step):
                hits = 0
                p1 = contour[i*frame_size+offset][0]
                p2 = contour[i*frame_size+frame_size+offset][0]
                solution = get_solution(p1, p2)
                if not len(solution):
                    continue
                for k in range(len(contour)):
                    checked_p = contour[k][0]
                    if solution[0] == 0:
                        if abs(solution[1] - get_x(checked_p)) < 2:
                            hits += 1
                    elif abs((get_y(checked_p) - solution[1])/solution[0] - get_x(checked_p)) < 2:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_solution = [p1, p2]
            i += 1
        p1 = best_solution[0]
        p2 = best_solution[1]
        x_distance = get_x(p2) - get_x(p1)
        y_distance = get_y(p2) - get_y(p1)
        if y_distance == 0:
            self.rotation = 0.0
        else:
            self.rotation = math.tan(x_distance / y_distance)
        if self.debug:
            img = cv2.imread(IMAGE_PATH)
            cropped_image = img[int(self.box[1]):int(self.box[3]), int(self.box[0]):int(self.box[2])]
            rotated = ndimage.rotate(cropped_image, -math.degrees(self.rotation))
            cv2.imshow("Puzzle rotated", rotated)
            cv2.imshow("Puzzle", cropped_image)
            cv2.waitKey()


    def _add_edges_types(self, contour):
        pass


def get_solution(p1, p2):
    coeff = np.array(([get_x(p1), 1], [get_x(p2), 1]))
    const = np.array(([get_y(p1), get_y(p2)]))
    solution = []
    try:
        solution = np.linalg.solve(coeff, const)
        return solution
    except LinAlgError as err:
        return solution

def get_x(p) -> int:
    return p[0]


def get_y(p) -> int:
    return p[1]