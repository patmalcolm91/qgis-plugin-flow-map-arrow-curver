"""
Implements a version of the algorithm described by Jenny et al
"""

from QBezier import *
from copy import deepcopy
from qgis.gui import QgsMessageBar
from qgis.core import *
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
        self.intermediatePointsCache = []  # holds a cache of the intermediate points to reduce redundant calculations

    def cacheIntermediatePoints(self, resolution):
        """
        Updates the intermediate points cache
        :param resolution: how many intermediate points to calculate
        :return: None
        :type resolution: int
        """
        self.intermediatePointsCache = self.getIntermediateCurvePoints(resolution)

    def getNearestIntermediatePoint(self, refPoint):
        """
        Returns the intermediate point (from the cache) nearest to the reference point
        :param refPoint: reference point. find intermediate point nearest to this point.
        :return: Point object
        :type refPoint: Point
        """
        points = self.intermediatePointsCache
        winner, winningDistance = None, None
        for point in points:
            dist = refPoint.distanceFrom(point)
            if winner is None or dist < winningDistance:
                winner = point
                winningDistance = dist
        return winner


class FlowMap(object):
    """
    An object which holds a set of Nodes and FlowLines along with various parameters.
    """
    def __init__(self, w_flows=1, w_nodes=0.5, w_antiTorsion=0.8, w_spring=1, w_angRes=3.75,
                 snapThreshold=0, bezierRes=15, alpha=4, beta=4, kShort=0.5, kLong=0.05, Cp=2.5, K=4, C=4,
                 constraintRectAspect=0.5):
        self.nodes = []  # type: list[Node]
        self.flowlines = []  # type: list[FlowLine]
        self.mapScale = 1  # type: int  # order of magnitude of the map. Used to scale values to avoid extreme values
        self.w_flows = w_flows  # type: float
        self.w_nodes = w_nodes  # type: float
        self.w_antiTorsion = w_antiTorsion  # type: float
        self.w_spring = w_spring  # type: float
        self.w_angRes = w_angRes  # type: float
        self.snapThreshold = snapThreshold  # type: float
        self.bezierRes = bezierRes  # type: int
        self.alpha = alpha  # type: int  # decay exponent for inter-flowline forces
        self.beta = beta  # type: int  # decay exponent for node-to-flow forces
        self.kShort = kShort  # type: float  # spring constant for short flowlines
        self.kLong = kLong  # type: float  # spring constant for long flowlines
        self.Cp = Cp  # type: float  # peripheral flow correction factor for spring constants
        self.K = K  # type: float  # angular resolution repulsion factor
        self.C = C  # type: float  # angular resolution clamping factor
        self.constraintRectAspect = constraintRectAspect

    def updateGeometryOnLayer(self, layer):
        """
        Updates the feature geometry on the layer based on the calculated curves.
        :param layer: layer which should be updated. Must contain FIDs matching the input layer (e.g. input layer)
        :return: None
        :type layer: QgsVectorLayer
        """
        geomDict = dict()
        for flowline in self.flowlines:
            geomDict[flowline.fid] = flowline.getWKT(self.bezierRes)
        layer.startEditing()
        for feature in layer.getFeatures():
            if feature.id() in geomDict:
                newGeom = QgsGeometry.fromWkt(geomDict[feature.id()])
                layer.dataProvider().changeGeometryValues({feature.id(): newGeom})
        layer.commitChanges()



    def getNodesFromLineLayer(self, layer):
        """
        Generates a list points containing all the unique nodes implied by the given line layer.
        :param layer: layer from which to get nodes
        :return: List(FlowMap.Point)
        :type layer: QgsVectorLayer
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
            flowline.cacheIntermediatePoints(self.bezierRes)
            self.flowlines.append(flowline)
            startNode.outflows.append(flowline)
            endNode.inflows.append(flowline)

    def calculateInterFlowLineForces(self, returnSumOfMagnitudes=False):
        """
        Calculates the flowline-against-flowline forces on each flowline.
        See section 3.1.1 of Jenny et al
        :param returnSumOfMagnitudes: if True, the sum of the magnitudes of the forces will also be returned.
        :return: list of forces in same order as self.flowlines
        :type returnSumOfMagnitudes: bool
        """
        forces = []  # type: list[Vector]
        sumsOfMagnitudes = []  # type: list[float]
        for flowline in self.flowlines:
            flPts = flowline.intermediatePointsCache
            Fflows = Vector(0, 0)
            n = 0
            sumOfMagnitudes = 0
            for pt in flPts:
                sumdw, sumw = Vector(0, 0), 0
                for other in self.flowlines:
                    if flowline == other or (
                            flowline.startNode == other.endNode and flowline.endNode == other.startNode):
                        continue  # skip identical or mirror images of this flowline
                    otherPts = other.intermediatePointsCache
                    for otherPt in otherPts:
                        dj = vectorFromPoints(otherPt, pt)
                        dj.scale(1.0/self.mapScale)
                        wj = math.pow(dj.getMagnitude(), -1*self.alpha)
                        sumdw += dj*wj
                        sumw += wj
                        n += 1
                Fp = sumdw.scale(1.0/sumw)
                sumOfMagnitudes += Fp.getMagnitude()
                Fflows += Fp
            Fflows.scale(float(self.mapScale)/n)
            forces.append(Fflows)
            sumOfMagnitudes *= float(self.mapScale/n)
            sumsOfMagnitudes.append(sumOfMagnitudes)
        if returnSumOfMagnitudes:
            return forces, sumsOfMagnitudes
        else:
            return forces

    def calculateNodeToFlowLineForces(self):
        """
        Calculates the force which nodes exert on each flowline.
        See section 3.1.2 of Jenny et al
        :return: list of forces in the same order as self.flowlines
        """
        forces = []  # type: list[Vector]
        for flowline in self.flowlines:
            sumdw, sumw = Vector(0, 0), 0
            for node in self.nodes:
                if node == flowline.startNode or node == flowline.endNode:
                    continue  # skip nodes to which we're connected
                nearestPoint = flowline.getNearestIntermediatePoint(node)
                dj = vectorFromPoints(node, nearestPoint)
                dj.scale(1.0 / self.mapScale)
                wj = math.pow(dj.getMagnitude(), -1 * self.beta)
                sumdw += dj * wj
                sumw += wj
            Fnodes = sumdw
            Fnodes.scale(float(self.mapScale)/sumw)
            forces.append(Fnodes)
        return forces

    def calculateAntiTorsionForces(self):
        """
        Calculates the anti-torsion forces on each flowline. This helps reduce asymmetry.
        See Section 3.1.3 of Jenny et al
        :return: list of anti-torsion forces in same order as self.flowlines
        """
        forces = []
        for flowline in self.flowlines:
            chordVector = vectorFromPoints(flowline.startNode, flowline.endNode)
            displacementVector = vectorFromPoints(flowline.p1, flowline.midPoint)
            force = displacementVector.projectionOnto(chordVector)
            forces.append(force)
        return forces

    def calculateSpringForces(self, interFlowLineForces, interFlowLineForceMagnitudes):
        """
        Calculates the spring forces on each flowline. This helps reduce curvature.
        See section 3.1.4 of Jenny et al
        :param interFlowLineForces: list of interFlowLine forces, needed for calculation of spring constants
        :param interFlowLineForceMagnitudes: list of magnitudes of interFlowLine force contributions
        :return: list of forces in same order as self.flowlines
        :type interFlowLineForces: list[Vector]
        :type interFlowLineForceMagnitudes: list[float]
        """
        # calculate the spring constants as specified in Jenny et al
        springLengths = [f.p1.distanceFrom(f.midPoint) for f in self.flowlines]
        B = [f.endNode.distanceFrom(f.startNode) for f in self.flowlines]
        Bmax = max(B)
        k = []
        for i, f in enumerate(self.flowlines):
            ki = (self.kLong - self.kShort)*(B[i]/Bmax) + self.kShort
            ki *= deepcopy(interFlowLineForces[i]).scale(1.0/interFlowLineForceMagnitudes[i]).getMagnitude()*self.Cp + 1
            k.append(ki)
        # calculate the spring forces
        forces = []
        for i, flowline in enumerate(self.flowlines):
            displacementVector = vectorFromPoints(flowline.p1, flowline.midPoint)
            force = displacementVector*k[i]
            forces.append(force)
        return forces

    def calculateAngleResForces(self):
        """
        Calculates the angular-resolution forces. This helps to increase angular resolution of flows at nodes.
        See Section 3.1.5 of Jenny et al
        :return: list of forces in the same order as self.flowlines
        """
        forces = []
        for flowline in self.flowlines:
            # Calculate start node forces
            Fs = 0
            startTan = vectorFromPoints(flowline.p0, flowline.p1)
            deltaStart = startTan.getDirection()
            ds = startTan.getMagnitude()
            sVectors = [o.getStartTangentVector() for o in flowline.startNode.outflows] +\
                       [i.getEndTangentVector().scale(-1) for i in flowline.startNode.inflows]
            for other in sVectors:
                if other == flowline:
                    continue  # skip this flowline
                deltai = deltaStart - other.getDirection()
                sign = 1 if deltai >= 0 else -1
                Fs += sign*math.exp(-1*self.K*(deltai**2))
            Fs *= ds
            # Calculate end node forces
            Fe = 0
            endTan = vectorFromPoints(flowline.p1, flowline.p2)
            deltaEnd = deepcopy(endTan).scale(-1).getDirection()
            de = endTan.getMagnitude()
            eVectors = [o.getStartTangentVector() for o in flowline.endNode.outflows] +\
                       [i.getEndTangentVector().scale(-1) for i in flowline.endNode.inflows]
            for other in eVectors:
                if other == flowline:
                    continue  # skip this flowline
                deltai = deltaEnd - other.getDirection()
                sign = 1 if deltai >= 0 else -1
                Fe += sign*math.exp(-1*self.K*(deltai**2))
            Fe *= de
            # Calculate resultant force vector
            FsVector = startTan.getPerpendicularVector().scale(-1*Fs)
            FeVector = endTan.getPerpendicularVector().scale(Fe)
            Fangres = FsVector + FeVector
            clampingValue = min(ds, de)/self.C
            if Fangres.getMagnitude() > clampingValue:
                Fangres.setMagnitude(clampingValue)
            forces.append(Fangres)
        return forces

    def applyForces(self, forces):
        """
        Applies forces to the control points.
        :param forces: list of forces to apply.
        :return: None
        :type forces: list[Vector]
        """
        for i, force in enumerate(forces):
            self.flowlines[i].p1 += force
            self.flowlines[i].cacheIntermediatePoints(self.bezierRes)

    def applyRectangleConstraints(self):
        """
        Constrains the control point to a rectangle along the arc's chord. The aspect of the rectangleis set using the
         parameter self.constraintRectAspect.
        See Section 3.2.1 of Jenny et al
        :return: None
        """
        for flowline in self.flowlines:
            displacementVector = vectorFromPoints(flowline.midPoint, flowline.p1)
            uConstr = vectorFromPoints(flowline.midPoint, flowline.endNode)  # the u-component constraint
            vConstr = Vector(uConstr.y, -1*uConstr.x).scale(self.constraintRectAspect)  # the v-component constraint
            uComp = displacementVector.projectionOnto(uConstr)
            uRatio = uComp.getMagnitude() / uConstr.getMagnitude()
            moved = False
            # If u-component constraint is exceeded, scale the vector by the appropriate amount
            if uRatio > 1:
                displacementVector.scale(1.0/uRatio)
                moved = True
            # If v-component constraint is exceeded, scale the vector by the appropriate amount
            vComp = displacementVector.projectionOnto(vConstr)
            vRatio = vComp.getMagnitude() / vConstr.getMagnitude()
            if vRatio > 1:
                displacementVector.scale(1.0/vRatio)
                moved = True
            if moved:
                flowline.p1 = flowline.midPoint + displacementVector
                flowline.cacheIntermediatePoints(self.bezierRes)


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
        w = 1 - float(i)/iterations
        # Calculate flowline-against-flowline forces
        flowline_forces, Fs = fm.calculateInterFlowLineForces(returnSumOfMagnitudes=True)
        # Calculate node-against-flowline forces
        node_forces = fm.calculateNodeToFlowLineForces()
        # Calculate anti-torsion forces
        antitorsion_forces = fm.calculateAntiTorsionForces()
        # Calculate spring forces
        spring_forces = fm.calculateSpringForces(flowline_forces, Fs)
        # Calculate angular-resolution-of-flowlines-around-nodes forces
        angRes_forces = fm.calculateAngleResForces()
        # Apply forces to arc control points
        resultantForces = []
        for i in range(len(fm.flowlines)):
            rForce = (flowline_forces[i]*fm.w_flows +
                      node_forces[i]*fm.w_nodes +
                      antitorsion_forces[i]*fm.w_antiTorsion +
                      spring_forces[i]*fm.w_spring)*w +\
                     (angRes_forces[i]*fm.w_angRes)*(w - w**2)
            resultantForces.append(rForce)
        fm.applyForces(resultantForces)
        # Apply rectangle constraint
        fm.applyRectangleConstraints()

    # Reduce intersections of flowlines with a common node

    # Move flows off of nodes

    # Update the geometry of the layer
    fm.updateGeometryOnLayer(lineLayer)
    iface.mapCanvas().refresh()


# TEST CODE ============================================================================================================
