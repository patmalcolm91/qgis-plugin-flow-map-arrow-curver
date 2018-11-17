# -*- coding: utf-8 -*-
"""
/***************************************************************************
 FlowMapArrowCurver
                                 A QGIS plugin
 Generates middle points for lines connecting OD flows.
                             -------------------
        begin                : 2018-11-16
        copyright            : (C) 2018 by Patrick Malcolm
        email                : patmalcolm91@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load FlowMapArrowCurver class from file FlowMapArrowCurver.

    :param iface: A QGIS interface instance.
    :type iface: QgisInterface
    """
    #
    from .flow_map_arrow_curver import FlowMapArrowCurver
    return FlowMapArrowCurver(iface)
