"""
Implements a version of the algorithm described by Jenny et al
"""

from QBezier import *
from copy import deepcopy
from qgis.gui import QgsMessageBar
import math


class Node(Point):
    """
    Node class which extends the point class
    """
    def __init__(self, x, y, radius=0):
        super(Node, self).__init__(x, y)
        self.inflows = []  # list of curves ending at this node
        self.outflows = []  # list of curves starting at this node
        self.radius = radius  # graphical radius of the node

    def __repr__(self):
        return "Node(" + str(self.x) + ", " + str(self.y) + ")"

    def fromPoint(self, point, radius=0):
        """
        Generates a node from a point object
        :param point: point object
        :param radius: graphical radius of node
        :return: node object
        :type point: Point
        """
        return Node(point.x, point.y, radius)


class FlowLine(QBezier):
    """
    Represents a flow line on the map.
    """
    def __init__(self, startNode, endNode, fid=None, width=0, constrainingRectangleWidthRatio=0.5):
        """
        :param startNode: start node
        :param endNode: end node
        :param width: graphical stroke width of flow line
        :param constrainingRectangleWidthRatio: ratio of width to length of constraining rectangle
        :type startNode: Node
        :type endNode: Node
        """
        midPoint = Point((startNode.x + endNode.x)/2.0, (startNode.y + endNode.y)/2.0)  # calculate midpoint
        super(FlowLine, self).__init__(startNode, midPoint, endNode)  # generate bezier curve
        self.midPoint = deepcopy(midPoint)  # save the midpoint
        self.width = width  # graphical width of line stroke
        self.constrainingRectangleWidth = endNode.distanceFrom(startNode)*constrainingRectangleWidthRatio
        self.fid = fid  # fid of the corresponding object
        self.startNode = self.p0
        self.endNode = self.p2


class FlowMap(object):
    """
    An object which holds a set of Nodes and FlowLines along with various parameters.
    """
    def __init__(self, snapThreshold=0, bezierRes=15, alpha=4):
        self.nodes = []  # type: list[Node]
        self.flowlines = []  # type: list[FlowLine]
        self.mapScale = 1  # type: int  # order of magnitude of the map. Used to scale values to avoid extreme values
        self.snapThreshold = snapThreshold  # type: float
        self.bezierRes = bezierRes  # type: int
        self.alpha = alpha  # type: int  # decay exponent for inter-flowline forces

    def getNodesFromLineLayer(self, layer):
        """
        Generates a list points containing all the unique nodes implied by the given line layer.
        :param layer: layer from which to get nodes
        :return: List(FlowMap.Point)
        """
        # first, loop through the features and add every start and end point
        for feature in layer.getFeatures():
            geom = feature.geometry().asPolyline()
            startNode = Node(geom[0].x(), geom[0].y())
            endNode = Node(geom[-1].x(), geom[-1].y())
            if startNode not in self.nodes:
                self.nodes.append(startNode)
            if endNode not in self.nodes:
                self.nodes.append(endNode)
        # now, loop back through and remove duplicates
        toRemove = []
        for i in range(len(self.nodes)-2):
            for j in range(i+1, len(self.nodes)):
                if self.nodes[i].distanceFrom(self.nodes[j]) <= self.snapThreshold:
                    # only add each duplicate once
                    if j not in toRemove:
                        toRemove.append(j)
        # now, loop backwards through the flagged duplicates and remove them
        toRemove.sort(reverse=True)
        for r in toRemove:
            self.nodes.pop(r)
        # return the remaining list
        return self.nodes

    def calculateMapScale(self):
        """
        Calculates the mean distance between the nodes in the list and sets it as self.mapScale
        Having this parameter correctly set helps prevent extremely large or small values in the force calculations
        :return: mean distance between the nodes in nodeList
        """
        distList = []
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    distList.append(i.distanceFrom(j))
        self.mapScale = sum(distList) / len(distList)

    def loadFlowLinesFromLayer(self, lineLayer):
        """
        Snaps line endpoints to nodes.
        :param lineLayer: line layer
        :type lineLayer: QgsVectorLayer
        :return: None
        """
        for feature in lineLayer.getFeatures():
            geom = feature.geometry().asPolyline()
            startPoint = Point(geom[0].x(), geom[0].y())
            endPoint = Point(geom[-1].x(), geom[-1].y())
            startNode, endNode = None, None
            for node in self.nodes:
                if startPoint.distanceFrom(node) <= self.snapThreshold:
                    startNode = node
                if endPoint.distanceFrom(node) <= self.snapThreshold:
                    endNode = node
            flowline = FlowLine(startNode, endNode, fid=feature.id())
            self.flowlines.append(flowline)
            startNode.outflows.append(flowline)
            endNode.inflows.append(flowline)

    def calculateInterFlowLineForces(self):
        """
        Calculates the flowline-against-flowline forces on each flowline.
        See section 3.1.1 of Jenny et al
        :return: list of forces in same order as self.flowlines
        """
        forces = []  # type: list[Vector]
        for flowline in self.flowlines:
            flPts = flowline.getIntermediateCurvePoints(self.bezierRes)
            Fflows = Vector(0, 0)
            n = 0
            for pt in flPts:
                sumdw, sumw = Vector(0, 0), 0
                for other in self.flowlines:
                    if flowline == other or (
                            flowline.startNode == other.endNode and flowline.endNode == other.startNode):
                        continue  # skip identical or mirror images of this flowline
                    otherPts = other.getIntermediateCurvePoints(self.bezierRes)
                    for otherPt in otherPts:
                        dj = vectorFromPoints(otherPt, pt)
                        dj.scale(1.0/self.mapScale)
                        wj = math.pow(dj.getMagnitude(), -1*self.alpha)
                        sumdw += dj*wj
                        sumw += wj
                        n += 1
                Fp = sumdw.scale(1/sumw)
                Fflows += Fp
            Fflows.scale(float(self.mapScale)/n)
            forces.append(Fflows)
        return forces


def run(iface, lineLayer, iterations, snapThreshold=0, bezierRes=15):
    """
    Runs the algorithm
    :return:
    """
    # Load the nodes and flows into a data structure
    fm = FlowMap(snapThreshold=snapThreshold, bezierRes=bezierRes)
    fm.getNodesFromLineLayer(lineLayer)
    fm.calculateMapScale()
    fm.loadFlowLinesFromLayer(lineLayer)
    # Iterate
    for i in range(iterations):
        # Calculate flowline-against-flowline forces
        flowline_forces = fm.calculateInterFlowLineForces()
        # Calculate node-against-flowline forces

        # Calculate anti-torsion forces

        # Calculate spring forces

        # Calculate angular-resolution-of-flowlines-around-nodes forces

        # Apply forces to arc control points

        # Apply rectangle constraint

        pass

    # Reduce intersections of flowlines with a common node

    # Move flows off of nodes

    iface.messageBar().pushMessage("Flow Map Arrow Curver", "Operation Complete", level=QgsMessageBar.INFO, duration=3)


# TEST CODE ============================================================================================================
