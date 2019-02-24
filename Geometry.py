"""
Holds classes and functions used by the algorithm.
"""

import math
from copy import deepcopy


class Point(object):
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

    def __eq__(self, other):
        result = self.x == other.x and self.y == other.y
        return result

    def distanceFrom(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Vector(object):
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
            result = Vector(self.x*other, self.y*other)
            return result

    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def scale(self, factor):
        """
        Scales the vector by the given factor.
        :param factor: factor by which to scale.
        :return: self
        """
        self.x *= factor
        self.y *= factor
        return self

    def getMagnitude(self):
        """
        Calculates the magnitude of the vector
        :return: Magnitude of the vector
        """
        try:
            return math.sqrt(self.x**2 + self.y**2)
        except:
            print(self)
            raise Exception("Numbers out of range.")

    def setMagnitude(self, magnitude):
        """
        Sets the magnitude of the vector, maintaining its direction.
        :param magnitude: the desired magnitude
        :return: self if successful, None if couldn't change due to zero magnitude.
        """
        m0 = self.getMagnitude()
        if m0 != 0:
            f = magnitude / m0
            self.scale(f)
            return self
        else:
            return None

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

    def projectionOnto(self, other):
        """
        Calculates the projection of this vector onto another.
        :param other: vector onto which to project
        :return: resultant vector
        :type other: Vector
        """
        dp = self.dotProduct(other)
        uv = Vector(other.x, other.y)
        uv.setMagnitude(dp/other.getMagnitude())
        return uv

    def getPerpendicularVector(self):
        """
        Calculates the vector perpendicular to the vector.
        :return: the perpendicular vector
        """
        result = Vector(self.y, self.x*-1)
        result.setMagnitude(1)
        return result


class Arc(object):
    """
    A 3-point arc class with methods to help interfacing with qgis geometry objects.
    """
    def __init__(self, fid, startPoint, endPoint):
        self.fid = fid  # corresponding object's FID
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.midPoint = Point((startPoint.x + endPoint.x)/2, (startPoint.y + endPoint.y)/2)  # straight-line midpoint
        self.controlPoint = deepcopy(self.midPoint)
        self.perpVector = self.getPerpendicularVector()

    def getWKT(self, asPolyline=False):
        """
        Generates the Well-Known-Text (WKT) string for this circular arc segment, or as a polyline if specified.
        :param asPolyline: if true, returns a 3-point polyline instead of an arc segment
        :return: WKT string
        """
        result = "LINESTRING(" if asPolyline else "CIRCULARSTRING("
        result += str(self.startPoint.x) + " " + str(self.startPoint.y) + ", "
        result += str(self.controlPoint.x) + " " + str(self.controlPoint.y) + ", "
        result += str(self.endPoint.x) + " " + str(self.endPoint.y) + ")"
        return result

    def getPerpendicularVector(self):
        """
        Calculates the vector perpendicular to the chord connecting the start and end point.
        :return: the perpendicular vector
        """
        chord = vectorFromPoints(self.startPoint, self.endPoint)
        return chord.getPerpendicularVector()


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


def distanceFromPointToLineSegment(point, lineStartPoint, lineEndPoint):
    """
    Calculates the shortest distance between the given point and the line segment between the given points.
    :param point: The point
    :param lineStartPoint: line segment start point
    :param lineEndPoint: line segment end point
    :return: shortest distance
    :type point: Point
    :type lineStartPoint: Point
    :type lineEndPoint: Point
    """
    segVector = vectorFromPoints(lineStartPoint, lineEndPoint)
    oVector = vectorFromPoints(lineStartPoint, point)
    theta = segVector.getDirection() - oVector.getDirection()
    if math.cos(theta) < 0:
        return point.distanceFrom(lineStartPoint)
    elif oVector.getMagnitude()*math.cos(theta) > segVector.getMagnitude():
        return point.distanceFrom(lineEndPoint)
    else:
        return abs(oVector.getMagnitude()*math.sin(theta))


def findIntersectionOfLines(start1, end1, start2, end2):
    """
    Returns the intersection of the lines defined by the given points, False if they are parallel, or True
    if the lines are coincident
    :param start1: start point of line 1
    :param end1: end point of line 1
    :param start2: start point of line 2
    :param end2: end point of line 2
    :return: intersection point of the lines, False if parallel, True if coincident
    :type start1: Point
    :type end1: Point
    :type start2: Point
    :type end2: Point
    """
    x1, y1, x2, y2 = start1.x, start1.y, end1.x, end1.y
    x3, y3, x4, y4 = start2.x, start2.y, end2.x, end2.y
    denominator = float(((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))
    if denominator == 0:  # if the two line segments are parallel or coincident
        if x1 == x2:  # if the first (and therefore also the second) line is vertical
            if max(y1, y2) > min(y3, y4) and min(y1, y2) < max(y3, y4) and x1 == x3:  # if lines are coincident vertical
                return True
            else:  # if the lines are parallel vertical
                return False
        else:  # the lines are not vertical, so check for coincidence
            m1 = (y2 - y1)/(x2 - x1)
            mBetween = (y3 - y2)/(x3 - x2)  # the slope between two points on different lines
            if m1 == mBetween:  # the lines are coincident
                return True
            else:  # the lines are parallel
                return False
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / denominator
    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    return Point(px, py)


def findIntersectionOfLineSegments(start1, end1, start2, end2):
    """
    Returns the intersection of the line segments defined by the given points, False if they do not intersect, or True
    if the lines are coincident
    :param start1: start point of line segment 1
    :param end1: end point of line segment 1
    :param start2: start point of line segment 2
    :param end2: end point of line segment 2
    :return: intersection point of the line segments, False if no intersection, True if coincident
    :type start1: Point
    :type end1: Point
    :type start2: Point
    :type end2: Point
    """
    x1, y1, x2, y2 = start1.x, start1.y, end1.x, end1.y
    x3, y3, x4, y4 = start2.x, start2.y, end2.x, end2.y
    denominator = float(((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))
    if denominator == 0:  # if the two line segments are parallel or coincident
        if x1 == x2:  # if the first (and therefore also the second) line is vertical
            if max(y1, y2) > min(y3, y4) and min(y1, y2) < max(y3, y4) and x1 == x3:  # if lines are coincident vertical
                return True
            else:  # if the lines are parallel vertical
                return False
        else:  # the lines are not vertical, so check for coincidence
            m1 = (y2 - y1)/(x2 - x1)
            mBetween = (y3 - y2)/(x3 - x2)  # the slope between two points on different lines
            if m1 == mBetween:  # the lines are coincident
                return True
            else:  # the lines are parallel
                return False
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / denominator
    if 0 <= t <= 1:
        u = -1*((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3)) / denominator
        if 0 <= u <= 1:
            px = x1 + t*(x2-x1)
            py = y1 + t*(y2-y1)
            return Point(px, py)
    return False


def distanceFromLineSegmentToLineSegment(start1, end1, start2, end2):
    """
    Calculates the shortest distance between two line segments defined by the given points.
    :param start1: start point of line segment 1
    :param end1: end point of line segment 1
    :param start2: start point of line segment 2
    :param end2: end point of line segment 2
    :return: shortest distance between line segments
    :type start1: Point
    :type end1: Point
    :type start2: Point
    :type end2: Point
    """
    intersect = findIntersectionOfLineSegments(start1, end1, start2, end2)
    # if the lines are coincident
    if intersect is True:
        return 0.0
    # if the lines don't intersect
    elif intersect is False:
        dist1 = distanceFromPointToLineSegment(start1, start2, end2)
        dist2 = distanceFromPointToLineSegment(end1, start2, end2)
        dist3 = distanceFromPointToLineSegment(start2, start1, end1)
        dist4 = distanceFromPointToLineSegment(end2, start1, end1)
        return min(dist1, dist2, dist3, dist4)
    # if intersect is a point, the lines intersect
    else:
        return 0.0


# TEST CODE ============================================================================================================

# p1, p2 = Point(1, 0), Point(5, 0)
# p3, p4 = Point(2, 4), Point(4, 2)
#
# print(findIntersectionOfLineSegments(p1, p2, p3, p4))
# print(distanceFromLineSegmentToLineSegment(p1, p2, p3, p4))
# print(distanceFromPointToLineSegment(p4, p1, p2))
# print(distanceFromPointToLineSegment(p2, p3, p4))
#
# v1 = Vector(4, 3)
# v2 = Vector(1, -1)
# print(v1.dotProduct(v2))
# print(v1)
# print(v2.getDirection()*180/math.pi)
# print(v1-v2)
# print(v2*3)
#
# p1, p2 = Point(0, 100), Point(100, 0)
# arc = Arc(1, p1, p2)
# perp = arc.getPerpendicularVector()
# arc.controlPoint += perp.scale(10)
# print(arc.getWKT())
#
# v1 = Vector(0, 20)
# v2 = Vector(3, 5)
# print(v2.projectionOnto(v1))
