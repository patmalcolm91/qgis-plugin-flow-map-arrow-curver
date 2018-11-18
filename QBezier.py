"""
Contains objects and functions for using quadratic bezier curves.
"""

from FlowMap import Point, Vector, vectorFromPoints
from qgis.core import *

class Node(Point):
    """
    Node class which extends the point class
    """
    def __init__(self, x, y):
        super(Node, self).__init__(x, y)
        self.inflows = []  # list of curves ending at this node
        self.outflows = []  # list of curves starting at this node

    def __repr__(self):
        return "Node(" + str(self.x) + ", " + str(self.y) + ")"


class QBezier(object):
    """
    A quadratic bezier class
    """
    def __init__(self, p0, p1, p2):
        """
        :param p0: start point
        :param p1: control point
        :param p2: end point
        :type p0: Point
        :type p1: Point
        :type p2: Point
        """
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def getIntermediateCurvePoints(self, num=1):
        """
        Returns list of num points that lie along the curve
        :param num: number of intermediate points to calculate
        :return: list[Point]
        """
        if num < 0:
            raise Exception("Can't retrieve negative number of intermediate points")
        tValues = [float(i+1)/(num+1) for i in range(num)]  # get evenly-spaced t values between 0 and 1
        result = []
        for t in tValues:
            # calculate the point using the quadratic bezier equation
            x = ((1-t)**2)*p0.x + 2*(1-t)*t*p1.x + t*t*p2.x
            y = ((1 - t) ** 2) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y
            result.append(Point(x, y))  # append the point to the list
        return result

    def getWKT(self, num=1):
        """
        Generates Well-Known-Text (WKT) for a linear-segmented representation of the curve with num intermediate points.
        :param num: number of intermediate points
        :return: WKT string
        """
        points = [self.p0] + self.getIntermediateCurvePoints(num) + [self.p2]
        result = "LINESTRING("
        for p in points:
            result += str(p.x) + " " + str(p.y) + ", "
        result = result[0:-2] + ")"
        return result

# TEST CODE ============================================================================================================

# n = Node(0, 4)
# print(n)
p0, p1, p2 = Point(10, 0), Point(0, 5), Point(0, 10)
qb = QBezier(p0, p1, p2)
print(qb.getWKT(5))
