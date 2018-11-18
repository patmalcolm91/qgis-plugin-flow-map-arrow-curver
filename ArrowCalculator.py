"""
Contains functions to calculate the arrow curves.
"""

import Geometry
from qgis.gui import QgsMessageBar
from qgis.core import *
from copy import deepcopy


def getNodesFromLineLayer(layer, threshold=0):
    """
    Generates a list points containing all the unique nodes implied by the given line layer.
    :param layer: layer from which to get nodes
    :param threshold: distance within which to consider two points to be the same
    :return: List(FlowMap.Point)
    """
    nodes = []  # hold the list of nodes
    # first, loop through the features and add every start and end point
    for feature in layer.getFeatures():
        geom = feature.geometry().asPolyline()
        startPoint = Geometry.Point(geom[0].x(), geom[0].y())
        endPoint = Geometry.Point(geom[-1].x(), geom[-1].y())
        if startPoint not in nodes:
            nodes.append(startPoint)
        if endPoint not in nodes:
            nodes.append(endPoint)
    # now, loop back through and remove duplicates
    toRemove = []
    for i in range(len(nodes)-2):
        for j in range(i+1, len(nodes)):
            if nodes[i].distanceFrom(nodes[j]) < threshold:
                # only add each duplicate once
                if j not in toRemove:
                    toRemove.append(j)
    # now, loop backwards through the flagged duplicates and remove them
    toRemove.sort(reverse=True)
    for r in toRemove:
        nodes.pop(r)
    # return the remaining list
    return nodes


def snapLinesToNodes(lineLayer, nodeList, threshold):
    """
    Snaps line endpoints to nodes.
    :param lineLayer: line layer
    :param nodeList: list of nodes
    :param threshold: threshold distance for snapping
    :type lineLayer: QgsVectorLayer
    :type nodeList: list(Point)
    :return: None
    """
    lineLayer.startEditing()
    for feature in lineLayer.getFeatures():
        geom = feature.geometry().asPolyline()
        startPoint = Geometry.Point(geom[0].x(), geom[0].y())
        endPoint = Geometry.Point(geom[-1].x(), geom[-1].y())
        for node in nodeList:
            nodePoint = QgsPoint(node.x, node.y)
            if startPoint.distanceFrom(node) < threshold:
                geom[0] = nodePoint
            if endPoint.distanceFrom(node) < threshold:
                geom[-1] = nodePoint
        newGeom = QgsGeometry.fromPolyline(geom)
        lineLayer.dataProvider().changeGeometryValues({feature.id(): newGeom})
    lineLayer.commitChanges()


def getArcsFromLayer(lineLayer):
    """
    Generates a list of Arc objects from the lines in the given layer.
    :param lineLayer: layer object
    :return: list[FlowMap.Arc]
    :type lineLayer QgsVectorLayer
    """
    arcs = []
    for feature in lineLayer.getFeatures():
        geom = feature.geometry().asPolyline()
        startPoint = Geometry.Point(geom[0].x(), geom[0].y())
        endPoint = Geometry.Point(geom[-1].x(), geom[-1].y())
        arcs.append(Geometry.Arc(feature.id(), startPoint, endPoint))
    return arcs


def getMeanDistanceBetweenNodes(nodeList):
    """
    Calculates the mean distance between the nodes in the list
    :param nodeList: list of nodes
    :return: mean distance between the nodes in nodeList
    :type nodeList: list[Geometry.Point]
    """
    distList = []
    for i in nodeList:
        for j in nodeList:
            if i != j:
                distList.append(i.distanceFrom(j))
    return sum(distList) / len(distList)


def run(iface, lineLayer, nodeThreshold, nodeSnap, repulsion, stiffness, springLength, stepSize,
        iterations, outputPolylines=False):
    """
    Runs the algorithm on the given line layer with the given settings.
    :param iface: the QGIS interface handle
    :param lineLayer: layer object
    :param nodeThreshold: distance within which two nodes are considered to be the same
    :param repulsion: repulsion factor between nodes and control points
    :param stiffness: stiffness of "spring" between control point and straight-line midpoint
    :param springLength: distance within which the stiffness has no effect
    :param stepSize: value to multiply by the force to get the displacement each iteration
    :param iterations: number of iterations
    :param outputPolylines: if true, output a 3-point polyline instead of a circular arc
    :return: None
    :type iface: QgisInterface
    :type lineLayer: QgsVectorLayer
    :type iterations: int
    """
    nodes = getNodesFromLineLayer(lineLayer, threshold=nodeThreshold)
    scale = getMeanDistanceBetweenNodes(nodes)
    if nodeSnap:
        snapLinesToNodes(lineLayer, nodes, nodeThreshold)
        iface.mapCanvas().refresh()
        iface.messageBar().pushMessage("Geometry edited", "Number of nodes snapped to: "+str(len(nodes)),
                                       level=QgsMessageBar.INFO, duration=2)
    arcs = getArcsFromLayer(lineLayer)
    for iteration in range(iterations):
        # Calculate the forces on each control point
        forces = [Geometry.Vector(0, 0) for a in arcs]
        for a, arc in enumerate(arcs):
            displacementVector = Geometry.vectorFromPoints(arc.midPoint, arc.controlPoint)  # displacement of the ctrl pt
            # calculate the spring force pulling the ctrl pt back to the midpt
            force = deepcopy(displacementVector)
            extension = springLength - displacementVector.getMagnitude()  # accounts for both compression and tension
            force.setMagnitude(extension)
            for node in nodes:
                dist = Geometry.vectorFromPoints(node, arc.controlPoint)
                dist.scale(1/scale)
                push = repulsion/(dist.getMagnitude()**2)
                dist.setMagnitude(scale*push)  # apply push in direction of dist
                force += dist
            forces[a] = force
        # Move the control points based on the forces acting on each
        for a, arc in enumerate(arcs):
            projForce = forces[a].projectionOnto(arc.perpVector)  # project the force onto the perpendicular
            # projForce *= stepSize
            arc.controlPoint += projForce
            if iteration % 10 == 0:
                print("force on fid " + str(arc.fid) + ": " + str(projForce))
    # iterations are done, so now we can write the geometry out
    lineLayer.startEditing()
    for arc in arcs:
        geom = QgsGeometry.fromWkt(arc.getWKT(asPolyline=outputPolylines))
        lineLayer.changeGeometry(arc.fid, geom)
    lineLayer.commitChanges()
    iface.mapCanvas().refresh()
    iface.messageBar().pushMessage("Flow Map Arrow Curver", "Operation Complete", level=QgsMessageBar.INFO, duration=2)
