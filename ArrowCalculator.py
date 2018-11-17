"""
Contains functions to calculate the arrow curves.
"""

import FlowMap


def run(iface, lineLayer, repulsion=1, stiffness=1):
    """
    Runs the algorithm on the given line layer with the given settings.
    :param iface: the QGIS interface handle
    :param lineLayer: layer object
    :return: None
    :type iface: QgisInterface
    :type lineLayer: QgisLayer
    """
    iface.legendInterface().setLayerVisible(lineLayer, False)
