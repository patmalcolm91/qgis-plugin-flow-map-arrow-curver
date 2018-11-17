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


def run(iface, lineLayer, nodeThreshold=0, nodeSnap=False, repulsion=1, stiffness=1):
    """
    Runs the algorithm on the given line layer with the given settings.
    :param iface: the QGIS interface handle
    :param lineLayer: layer object
    :param nodeThreshold: distance within which two nodes are considered to be the same
    :param repulsion: repulsion factor between nodes and control points
    :param stiffness: stiffness of "spring" between control point and straight-line midpoint
    :return: None
    :type iface: QgisInterface
    :type lineLayer: QgsVectorLayer
    """
    nodes = getNodesFromLineLayer(lineLayer, threshold=nodeThreshold)
    if nodeSnap:
        snapLinesToNodes(lineLayer, nodes, nodeThreshold)
        iface.mapCanvas().refresh()
        iface.messageBar().pushMessage("Geometry edited", "Number of nodes found: "+str(len(nodes)),
                                       level=QgsMessageBar.INFO, duration=2)
