from enum import Enum
import numpy as np
import math
import cv2
from numpy.linalg import LinAlgError
from scipy import ndimage


class EdgeType(Enum):
    FLAT = 0
    FEMALE = 1
    MALE = 2
    UNDEFINED = 3


class Puzzle:

    def __init__(self, mask, box, path):
        self.data = []
        self.edges_types = {
            "left": EdgeType.UNDEFINED,
            "right": EdgeType.UNDEFINED,
            "up": EdgeType.UNDEFINED,
            "down": EdgeType.UNDEFINED}
        self.rotation = 0.0
        self.box = box.tensor.numpy()[0]
        self.center = box.get_centers().numpy()[0]
        self.rotated_image = []
        self.position = (None, None) # row, column
        self.path = path

        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contour, key=cv2.contourArea)    
        self._rotate_to_right_angle(contour)
        self._add_edges_types(mask)
    

    def set_position(self, coords):
        self.position = coords


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
            self.fix_rotation()


    def _add_edges_types(self, mask):
        cropped_image = mask[int(self.box[1]):int(self.box[3]), int(self.box[0]):int(self.box[2])]
        rotated = ndimage.rotate(cropped_image, -math.degrees(self.rotation), reshape=False)

        counts = np.sum(rotated == 1, axis=0)
        max_y = counts.max()

        counts = np.sum(rotated == 1, axis=1)
        max_x = counts.max()

        hit_up = self._check_hits(rotated, True, False)
        hit_down = self._check_hits(rotated, True, True)
        hit_left = self._check_hits(rotated, False, False)
        hit_right = self._check_hits(rotated, False, True)
        
        self._change_puzzle_type(hit_up/max_x, "up")
        self._change_puzzle_type(hit_down/max_x, "down")
        self._change_puzzle_type(hit_left/max_y, "left")
        self._change_puzzle_type(hit_right/max_y, "right")

        img = cv2.imread(self.path)
        cropped_image = img[int(self.box[1]):int(self.box[3]), int(self.box[0]):int(self.box[2])]
        self.rotated_image = ndimage.rotate(cropped_image, -math.degrees(self.rotation), reshape=False)

        # print(self.edges_types)
        # self.show()

    

    def _check_hits(self, img, if_x, if_end):
        height = int(self.box[3] - self.box[1])
        width = int(self.box[2] - self.box[0])
        range_max = width if if_x else height
        range_max2 = height if if_x else width
        hit = 0
        const_offset = int(range_max2/15)
        offset = const_offset
        
        while hit == 0:
            idx = range_max2 - offset if if_end else offset

            arr = img[idx, :] if if_x else img[:, idx]
            hit = sum(arr)
            changes = np.diff(np.concatenate(([0], arr, [0])))

            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            lengths = ends - starts
            offset += const_offset
        return hit


    def _change_puzzle_type(self, ratio, side):
        THRESHOLD = 0.4
        THRESHOLD_EDGE = 0.8

        if ratio > THRESHOLD_EDGE:
            self.edges_types[side] = EdgeType.FLAT
        else:
            self.edges_types[side] = EdgeType.FEMALE if ratio > THRESHOLD else EdgeType.MALE


    def show(self):
        cv2.imshow("Puzzle rotated", self.rotated_image)
        cv2.waitKey()
    

    def compare(self, puzzle, side):
        best_score = 0.0

        opposite = {
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }


        for i in range(4):

            other_side = opposite[side]
            if self.edges_types[side] == puzzle.edges_types[other_side]:
                continue

            edge1 = self._extract_edge(self.rotated_image, side)
            edge2 = puzzle._extract_edge(puzzle.rotated_image, other_side)
            edge2 = np.flip(edge2, axis=0)
            min_len = min(len(edge1), len(edge2))
            edge1 = edge1[:min_len]
            edge2 = edge2[:min_len]

            diff = np.linalg.norm(
                edge1.astype(np.float32) -
                edge2.astype(np.float32),
                axis=1
            )

            similarity = 1.0 - np.mean(diff) / 255.0
            best_score = max(best_score, similarity)
            if best_score > 0.9:
                break
            puzzle.rotate(90)

        return best_score
    

    def _extract_edge(self, img, side, thickness=5):
        h, w = img.shape[:2]

        if side == "left":
            edge = img[:, :thickness]
        elif side == "right":
            edge = img[:, w-thickness:w]
        elif side == "up":
            edge = img[:thickness, :]
        else:
            edge = img[h-thickness:h, :]

        return np.mean(edge, axis=1 if side in ["left", "right"] else 0)


    def rotate(self, rotation): # rotation in degrees
        times = int(rotation/math.pi/2)
        self.rotation += rotation
        self.fix_rotation()
        self.rotated_image = ndimage.rotate(self.rotated_image, -math.degrees(self.rotation), reshape=False)

        for _ in range(times):
            new_edges_types = {
                "left": self.edges_types["down"],
                "right": self.edges_types["up"],
                "up": self.edges_types["left"],
                "down": self.edges_types["right"]}
            self.edges_types = new_edges_types
    

    def fix_rotation(self):
        self.rotation = (self.rotation + math.pi) % (2 * math.pi) - math.pi



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