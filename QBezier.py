"""
Contains objects and functions for using quadratic bezier curves.
"""

from FlowMap import Point, Vector, vectorFromPoints


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


# TEST CODE ============================================================================================================

n = Node(0, 4)
print(n)