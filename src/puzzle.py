from enum import Enum


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
