"""
Holds classes and functions used by the algorithm.
"""

import math
from copy import deepcopy


class Point:
    """
    A 2D point class.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    def distanceFrom(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Vector:
    """
    A 2D vector class.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Vector<" + str(self.x) + ", " + str(self.y) + ">"

    def __mul__(self, other):
        if hasattr(other, "x") and hasattr(other, "y"):
            return self.dotProduct(other)
        else:
            result = deepcopy(self)
            result.scale(other)
            return result

    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def scale(self, factor):
        """
        Scales the vector by the given factor.
        :param factor: factor by which to scale.
        :return: None
        """
        self.x *= factor
        self.y *= factor

    def getMagnitude(self):
        """
        Calculates the magnitude of the vector
        :return: Magnitude of the vector
        """
        return math.sqrt(self.x**2 + self.y**2)

    def setMagnitude(self, magnitude):
        """
        Sets the magnitude of the vector, maintaining its direction.
        :param magnitude: the desired magnitude
        :return: True if successful, False if couldn't change due to zero magnitude.
        """
        m0 = self.getMagnitude()
        if m0 != 0:
            f = magnitude / m0
            self.scale(f)
            return True
        else:
            return False

    def normalize(self):
        """
        Sets the magnitude of the vector to 1.
        :return: True if successful, False if couldn't change due to zero magnitude.
        """
        return self.setMagnitude(1)

    def getDirection(self):
        """
        Calculates the geometric direction of the vector (in radians, starting to the right and going CCW)
        :return: the geometric direction in radians
        """
        angle = math.atan2(self.y, self.x)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def dotProduct(self, vector):
        """
        Calculates the dot product of this vector with another vector.
        :param vector: the vector to dot with
        :return: scalar result of the dot product
        :type vector: Vector
        """
        return self.x * vector.x + self.y * vector.y


class Arc:
    """
    A 3-point arc class with methods to help interfacing with qgis geometry objects.
    """
    def __init__(self, fid, startPoint, endPoint):
        self.fid = fid  # corresponding object's FID
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.midPoint = Point((startPoint.x + endPoint.x)/2, (startPoint.y + endPoint.y)/2)  # straight-line midpoint
        self.controlPoint = deepcopy(self.midPoint)

    def getWKT(self):
        """
        Generates the Well-Known-Text (WKT) string for this circular arc segment.
        :return:
        """
        result = "CIRCULARSTRING("
        result += str(self.startPoint.x) + " " + str(self.startPoint.y) + ", "
        result += str(self.controlPoint.x) + " " + str(self.controlPoint.y) + ", "
        result += str(self.endPoint.x) + " " + str(self.endPoint.y) + ")"
        return result


def vectorFromPoints(p1, p2):
    """
    Calculates the vector between the two points
    :param p1: point 1
    :param p2: point 2
    :return: Vector from point 1 to point 2
    :type p1: Point
    :type p2: Point
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return Vector(dx, dy)


# TEST CODE ============================================================================================================

# v1 = Vector(4, 3)
# v2 = Vector(1, -1)
# print(v1.dotProduct(v2))
# print(v1)
# print(v2.getDirection()*180/math.pi)
# print(v1-v2)
# print(v2*3)
