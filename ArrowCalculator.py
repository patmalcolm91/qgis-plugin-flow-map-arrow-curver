"""
Contains functions to calculate the arrow curves.
"""

import FlowMap
from qgis.gui import QgsMessageBar
from qgis.core import *


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
        startPoint = FlowMap.Point(geom[0].x(), geom[0].y())
        endPoint = FlowMap.Point(geom[-1].x(), geom[-1].y())
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
        startPoint = FlowMap.Point(geom[0].x(), geom[0].y())
        endPoint = FlowMap.Point(geom[-1].x(), geom[-1].y())
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
    :return: list(FlowMap.Arc)
    :type lineLayer QgsVectorLayer
    """
    arcs = []
    for feature in lineLayer.getFeatures():
        geom = feature.geometry().asPolyline()
        startPoint = FlowMap.Point(geom[0].x(), geom[0].y())
        endPoint = FlowMap.Point(geom[-1].x(), geom[-1].y())
        arcs.append(FlowMap.Arc(feature.id(), startPoint, endPoint))
    return arcs


def run(iface, lineLayer, nodeThreshold=0, nodeSnap=False, repulsion=50, stiffness=1, springLength=50, stepSize=5, iterations=10):
    """
    Runs the algorithm on the given line layer with the given settings.
    :param iface: the QGIS interface handle
    :param lineLayer: layer object
    :param nodeThreshold: distance within which two nodes are considered to be the same
    :param repulsion: repulsion factor between nodes and control points
    :param stiffness: stiffness of "spring" between control point and straight-line midpoint
    :param springLength: distance within which the stiffness has no effect
    :return: None
    :type iface: QgisInterface
    :type lineLayer: QgsVectorLayer
    """
    nodes = getNodesFromLineLayer(lineLayer, threshold=nodeThreshold)
    if nodeSnap:
        snapLinesToNodes(lineLayer, nodes, nodeThreshold)
        iface.mapCanvas().refresh()
        iface.messageBar().pushMessage("Geometry edited", "Number of nodes snapped to: "+str(len(nodes)),
                                       level=QgsMessageBar.INFO, duration=2)
    arcs = getArcsFromLayer(lineLayer)
    for iteration in range(iterations):
        # Calculate the forces on each control point
        forces = []
        for arc in arcs:
            displacementVector = FlowMap.vectorFromPoints(arc.midPoint, arc.controlPoint)  # displacement of the ctrl pt
            force = displacementVector*-1*stiffness  # the spring force pulling the ctrl pt back to the midpt
            if displacementVector.getMagnitude() < springLength:
                force *= -1  # if we're within the spring length, the force should go backwards
            for node in nodes:
                dist = FlowMap.vectorFromPoints(node, arc.controlPoint)
                push = repulsion/(dist.getMagnitude()**2)
                dist.setMagnitude(push)
                force += dist
            forces.append(force)
        # Move the control points based on the forces acting on each
        for a, arc in enumerate(arcs):
            arc.controlPoint += forces[a]*stepSize
    # iterations are done, so now we can write the geometry out
    lineLayer.startEditing()
    for arc in arcs:
        geom = QgsGeometry.fromWkt(arc.getWKT())
        lineLayer.changeGeometry(arc.fid, geom)
    lineLayer.commitChanges()
    iface.mapCanvas().refresh()
    iface.messageBar().pushMessage("Flow Map Arrow Curver", "Operation Complete", level=QgsMessageBar.INFO, duration=2)
