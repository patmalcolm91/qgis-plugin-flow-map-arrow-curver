"""
Implements a version of the algorithm described by Jenny et al
"""

import Geometry
from QBezier import *
from copy import deepcopy
from qgis.gui import QgsMessageBar
from PyQt4.QtGui import QProgressDialog, QProgressBar
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
        self.locked = False  # whether or not the position of this flowline is locked, if True, forces won't affect it

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

    def shortestDistanceFromPoint(self, otherPoint):
        """
        Returns the shortest distance from the linear-segmented version of the flowline to a given point
        :return: shortest distance from point to flowline
        :type otherPoint: Point
        """
        shortestDist = None
        pointList = [self.startNode] + self.intermediatePointsCache + [self.endNode]
        for i in range(len(pointList)-1):
            distToSegment = Geometry.distanceFromPointToLineSegment(otherPoint, pointList[i], pointList[i+1])
            if shortestDist is None or distToSegment < shortestDist:
                if distToSegment == 0:
                    return 0
                shortestDist = distToSegment
        return shortestDist

    def interesectsFlowLine(self, otherFlowLine):
        """
        Determines whether this flowline intersects another, ignoring endpoints
        :param otherFlowLine: the flowline to check against
        :return: True if they intersect, False if not or they intersect at an endpoint
        :type otherFlowLine: FlowLine
        """
        for i in range(len(self.intermediatePointsCache)-1):
            p1, p2 = self.intermediatePointsCache[i], self.intermediatePointsCache[i+1]
            for j in range(len(otherFlowLine.intermediatePointsCache)-1):
                p3, p4 = otherFlowLine.intermediatePointsCache[j], otherFlowLine.intermediatePointsCache[j+1]
                intersect = Geometry.findIntersectionOfLineSegments(p1, p2, p3, p4)
                if intersect is True or intersect is not False:
                    return True
        return False

    def constrainPointInsideRectangle(self, point, referencePoint=None):
        """
        Constrains a point inside the constraining rectangle based on the path from the reference point.
        :param point: the point which to constrain into the rectangle
        :param referencePoint: from where to measure the displacement. Default (None) uses midpoint
        :return: point if inside rectangle, else a point along the path from referencePoint to point which is.
        :type point: Point
        :type referencePoint: Point
        """
        if referencePoint is None:
            referencePoint = self.midPoint
        displacementVector = vectorFromPoints(referencePoint, point)
        uConstr = vectorFromPoints(self.midPoint, self.endNode)  # the u-component constraint
        vConstr = Vector(uConstr.y, -1 * uConstr.x).setMagnitude(self.constrainingRectangleWidth/2.0)  # the v-constr.
        uComp = displacementVector.projectionOnto(uConstr)
        uRatio = uComp.getMagnitude() / uConstr.getMagnitude()
        moved = False
        # If u-component constraint is exceeded, scale the vector by the appropriate amount
        if uRatio > 1:
            displacementVector.scale(1.0 / uRatio)
            moved = True
        # If v-component constraint is exceeded, scale the vector by the appropriate amount
        vComp = displacementVector.projectionOnto(vConstr)
        vRatio = vComp.getMagnitude() / vConstr.getMagnitude()
        if vRatio > 1:
            displacementVector.scale(1.0 / vRatio)
            moved = True
        if not moved:
            return point
        else:
            return point+displacementVector


class FlowMap(object):
    """
    An object which holds a set of Nodes and FlowLines along with various parameters.
    """
    def __init__(self, w_flows=1, w_nodes=0.5, w_antiTorsion=0.8, w_spring=1, w_angRes=3.75,
                 snapThreshold=0, bezierRes=15, alpha=4, beta=4, kShort=0.5, kLong=0.05, Cp=2.5, K=4, C=4,
                 constraintRectAspect=0.5, nodeBuffer=0):
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
        self.nodeBuffer = nodeBuffer

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

    def getNearestNode(self, point):
        """
        returns the node which is nearest to the given point
        :param point: the point
        :return: the relevant node
        :type point: Point
        """
        winner = None
        winningDist = None
        for node in self.nodes:
            dist = node.distanceFrom(point)
            if winningDist is None or dist < winningDist:
                winner = node
                winningDist = dist
            if dist == 0:
                break
        return winner

    def getNodeRadiiFromLayer(self, nodeLayer, expression):
        """
        Reads a point layer and evaluates an expression to generate a list of graphical node radii.
        :param nodeLayer: point layer containing nodes.
        :param expression: expression to be evaluated to get the radii.
        :return: None
        :type nodeLayer: QgsVectorLayer
        :type expression: QString
        """
        context = QgsExpressionContext()
        scope = QgsExpressionContextScope()
        context.appendScope(scope)
        for feat in nodeLayer.getFeatures():
            scope.setFeature(feat)
            exp = QgsExpression(expression)
            val = exp.evaluate(context)
            radius = 0 if val is None else val
            geom = feat.geometry().asPoint()
            self.getNearestNode(Point(geom.x(), geom.y())).radius = radius

    def averageNodeRadius(self):
        """
        Returns the average of the node radii
        :return: average radius of all nodes
        """
        sum = 0
        for node in self.nodes:
            sum += node.radius
        return float(sum) / len(self.nodes)

    def getLineWidthFromLayer(self, lineLayer, expression):
        """
        Reads a point layer and evaluates an expression to generate a list of graphical line widths.
        :param lineLayer: line layer containing nodes.
        :param expression: expression to be evaluated to get the radii.
        :return: None
        :type lineLayer: QgsVectorLayer
        :type expression: QString
        """
        context = QgsExpressionContext()
        scope = QgsExpressionContextScope()
        context.appendScope(scope)
        for feat in lineLayer.getFeatures():
            scope.setFeature(feat)
            exp = QgsExpression(expression)
            val = exp.evaluate(context)
            width = 0 if val is None else val
            # Find the corresponding edge and assign its width
            for flowline in self.flowlines:
                if flowline.fid == feat.id():
                    flowline.width = width
                    break

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
            if startPoint == endPoint:
                raise Warning("Flowline with zero length detected. Skipping")
                continue
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
            if not self.flowlines[i].locked:
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

    def flowLineOverlapsNodes(self, flowline):
        """
        Checks if the given flowline overlaps any nodes.
        :param flowline: the flowline to check
        :return: True if overlaps, False if not
        """
        for node in self.nodes:
            reqdDist = node.radius + self.nodeBuffer + flowline.width
            if flowline.shortestDistanceFromPoint(node) < reqdDist:
                return True
        return False

    def getFlowLinesOverlappingNodes(self):
        """
        Returns a list of flowlines which overlap a node.
        :return:
        """
        result = []
        for flowline in self.flowlines:
            for node in self.nodes:
                reqdDist = node.radius + self.nodeBuffer + flowline.width
                if flowline.shortestDistanceFromPoint(node) < reqdDist:
                    result.append(flowline)
                    continue
        return result

    def reduceFlowLineIntersections(self):
        """
        Reduces intersections between flowlines which share a common node.
        See Section 3.2.2 of Jenny et al
        :return: None
        """
        for node in self.nodes:
            inAndOutFlows = node.inflows + node.outflows  # type: list[FlowLine]
            for i in range(len(inAndOutFlows)-1):
                if inAndOutFlows[i].locked:
                    continue
                for j in range(i+1, len(inAndOutFlows)):
                    if inAndOutFlows[j].locked:
                        continue
                    if inAndOutFlows[i].interesectsFlowLine(inAndOutFlows[j]):
                        pointM = inAndOutFlows[i].p1
                        pointN = inAndOutFlows[j].p1
                        # determine point A
                        if inAndOutFlows[i] in node.outflows:
                            pointA = inAndOutFlows[i].endNode
                        elif inAndOutFlows[i] in node.inflows:
                            pointA = inAndOutFlows[i].startNode
                        else:
                            raise KeyError("Couldn't find flow belonging to node.")
                        # determine point B
                        if inAndOutFlows[j] in node.outflows:
                            pointB = inAndOutFlows[j].endNode
                        elif inAndOutFlows[j] in node.inflows:
                            pointB = inAndOutFlows[j].startNode
                        else:
                            raise KeyError("Couldn't find flow belonging to node.")
                        # Make sure the other points on the two flows aren't the same
                        if pointA == pointB:
                            continue
                        Mbar = Geometry.findIntersectionOfLines(pointM, pointA, node, pointN)
                        Nbar = Geometry.findIntersectionOfLines(pointN, pointB, node, pointM)
                        Mbar = inAndOutFlows[i].constrainPointInsideRectangle(Mbar, pointM)
                        Nbar = inAndOutFlows[j].constrainPointInsideRectangle(Nbar, pointN)
                        inAndOutFlows[i].p1 = Mbar
                        inAndOutFlows[j].p1 = Nbar
                        inAndOutFlows[i].cacheIntermediatePoints(self.bezierRes)
                        inAndOutFlows[j].cacheIntermediatePoints(self.bezierRes)


    def pointIsWithinFlowLineRectangle(self, point, flowline):
        """
        Checks if a point is within the the constraint rectangle of the given flowline
        :param point: the point to check
        :param flowline: the flowline for which to calculate the constraining rectangle
        :return: True if point within rectangle, else False
        :type point: Point
        :type flowline: FlowLine
        """
        displacementVector = vectorFromPoints(flowline.midPoint, point)
        uConstr = vectorFromPoints(flowline.midPoint, flowline.endNode)  # the u-component constraint
        vConstr = Vector(uConstr.y, -1 * uConstr.x).scale(self.constraintRectAspect)  # the v-component constraint
        uComp = displacementVector.projectionOnto(uConstr)
        vComp = displacementVector.projectionOnto(vConstr)
        uRatio = uComp.getMagnitude() / uConstr.getMagnitude()
        vRatio = vComp.getMagnitude() / vConstr.getMagnitude()
        if uRatio <= 1 or vRatio <= 1:
            return True
        else:
            return False

    def moveFlowLineOffNodes(self, flowline):
        """
        Moves specified flowline away from nodes which they intersect by moving control point in an Archimedean spiral.
        See Section 3.2.3 of Jenny et al
        :param flowline: the flowline on which to act
        :return: True if successful, False if no candidate point was suitable and flowline remains unchanged
        :type flowline: FlowLine
        """
        trialCurve = deepcopy(flowline)  # work on a copy of the curve so we can safely modify its geometry
        # the archimedean spiral radial change constant. Per Jenny et al, set to the minimum buffer distance
        rho = self.averageNodeRadius() + flowline.width + self.nodeBuffer
        spacing = rho  # spacing along the curve between each candidate point. Set to the same as rho, per Jenny et al
        outOfBoundsPointAngle = 0
        lastPointWasInBounds = True  # whether or not the last point checked was in bounds
        step = 1  # iteration step
        while True:
            # Calculate the candidate point on the spiral
            theta = math.sqrt(2.0*spacing*step/rho)
            if theta >= outOfBoundsPointAngle + 2*math.pi:
                return False
            displacementVector = Vector(rho*theta*math.cos(theta), rho*theta*math.sin(theta))
            trialCurve.p1 = flowline.p1 + displacementVector
            if self.pointIsWithinFlowLineRectangle(trialCurve.p1, trialCurve):
                trialCurve.cacheIntermediatePoints(self.bezierRes)
                if not self.flowLineOverlapsNodes(trialCurve):
                    flowline.p1 = trialCurve.p1
                    flowline.cacheIntermediatePoints()
                    flowline.locked = True
                    return True
                lastPointWasInBounds = True
            elif lastPointWasInBounds == True:
                lastPointWasInBounds = False
                outOfBoundsPointAngle = theta
            step += 1


def run(iface, lineLayer, nodeLayer, nodeRadiiExpr="0", lineWidthExpr="1", iterations=100, w_flows=1, w_nodes=0.5,
        w_antiTorsion=0.8, w_spring=1, w_angRes=3.75, snapThreshold=0, bezierRes=15, alpha=4, beta=4, kShort=0.5,
        kLong=0.05, Cp=2.5, K=4, C=4, constraintRectAspect=0.5, nodeBuffer=0):
    """
    Runs the algorithm
    :return:
    """
    # Generate a progress bar
    progDialog = QProgressDialog()
    progDialog.setWindowTitle("Progress")
    progDialog.setLabelText("Generating flowmap from layers")
    progBar = QProgressBar(progDialog)
    progBar.setTextVisible(True)
    progBar.setValue(0)
    progDialog.setBar(progBar)
    progDialog.setMinimumWidth(300)
    progBar.setMaximum(iterations)
    progDialog.show()
    # Load the nodes and flows into a data structure
    fm = FlowMap(w_flows=w_flows, w_nodes=w_nodes, w_antiTorsion=w_antiTorsion, w_spring=w_spring, w_angRes=w_angRes,
                 snapThreshold=snapThreshold, bezierRes=bezierRes, alpha=alpha, beta=beta, kShort=kShort, kLong=kLong,
                 Cp=Cp, K=K, C=C, constraintRectAspect=constraintRectAspect, nodeBuffer=nodeBuffer)
    fm.getNodesFromLineLayer(lineLayer)
    if nodeLayer is not None:
        fm.getNodeRadiiFromLayer(nodeLayer, nodeRadiiExpr)
    else:
        fm.nodeRadii = [0 for node in fm.nodes]
    fm.getLineWidthFromLayer(lineLayer, lineWidthExpr)
    fm.calculateMapScale()
    fm.loadFlowLinesFromLayer(lineLayer)
    # Iterate
    j = 0  # used in node collision avoidance algorithm
    progDialog.setLabelText("Iterating")
    for i in range(iterations):
        progBar.setValue(i)
        if progDialog.wasCanceled():
            break
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
        fm.reduceFlowLineIntersections()
        # Move flows off of nodes
        if i > 0.1*iterations and j <= 0:
            N = fm.getFlowLinesOverlappingNodes()
            j = (iterations - i)/(len(N) + 1)/2
            n = math.ceil(float(len(N))/(iterations - i - 1))
            movedFlows = 0  # number of flows which have been moved. Iterate until this equals n
            for flowline in N:
                movedFlows += 1 if fm.moveFlowLineOffNodes(flowline) else 0
                if movedFlows >= n:
                    break
        else:
            j -= 1

    progDialog.setLabelText("Updating Geometry")
    # Update the geometry of the layer
    fm.updateGeometryOnLayer(lineLayer)
    iface.mapCanvas().refresh()
    # close the dialog
    progDialog.close()


# TEST CODE ============================================================================================================
