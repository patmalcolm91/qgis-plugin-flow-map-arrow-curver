# -*- coding: utf-8 -*-
"""
/***************************************************************************
 FlowMapArrowCurver
                                 A QGIS plugin
 Generates middle points for lines connecting OD flows.
                              -------------------
        begin                : 2018-11-16
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Patrick Malcolm
        email                : patmalcolm91@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from flow_map_arrow_curver_dialog import FlowMapArrowCurverDialog
import os.path
import JennyAlgorithm
from qgis.core import *
from qgis.gui import QgsMessageBar, QgsFieldExpressionWidget


class FlowMapArrowCurver:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgisInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'FlowMapArrowCurver_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Flow Map Curved Arrow Calculator')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'FlowMapArrowCurver')
        self.toolbar.setObjectName(u'FlowMapArrowCurver')

        # Custom properties
        self.layers = self.iface.legendInterface().layers()
        self.lineLayerList = []
        self.lineLayerIndexMap = dict()
        self.pointLayerList = []
        self.pointLayerIndexMap = dict()

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('FlowMapArrowCurver', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference
        self.dlg = FlowMapArrowCurverDialog()

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/FlowMapArrowCurver/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Flow Map Curved Arrow Calculator'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Flow Map Curved Arrow Calculator'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def refreshLayerLists(self):
        """
        Re-generates the layer lists.
        :return: None
        """
        self.layers = self.iface.legendInterface().layers()
        self.lineLayerIndexMap = dict()
        self.pointLayerIndexMap = dict()
        self.lineLayerList = []  # holds the filtered layer names
        for i, layer in enumerate(self.layers):
            if layer.geometryType() == 0:  # 0: point, 1: line
                self.pointLayerIndexMap[len(self.pointLayerList)] = i  # put the index pair in the dictionary
                self.pointLayerList.append(layer.name())  # add the layer name to the list
            if layer.geometryType() == 1:  # 0: point, 1: line
                self.lineLayerIndexMap[len(self.lineLayerList)] = i  # put the index pair in the dictionary
                self.lineLayerList.append(layer.name())  # add the layer name to the list


    def updateFieldExpressionWidgets(self, index):
        """
        Updates the field expression widgets when the layer changes
        :return: None
        """
        # Get the selected layer
        # index = self.lineLayerIndexMap[self.dlg.lineLayerChooser.currentIndex()]
        selectedLayer = self.layers[index]  # type: QgsVectorLayer
        self.dlg.lineWidthExpressionWidget.setLayer(selectedLayer)
        self.dlg.lineWidthExpressionWidget.setExpression("1")

    def nodeLayerEnabledStateChanged(self, state):
        """
        callback function for when the node layer enabled checkbox changes state.
        :param state:
        :return:
        """
        if state == 0:
            self.dlg.nodeLayerChooser.clear()
            self.dlg.nodeLayerChooser.setEnabled(False)
        elif state == 2:
            self.dlg.nodeLayerChooser.addItems(self.pointLayerList)
            self.dlg.nodeLayerChooser.setEnabled(True)

    def run(self):
        """Run method that performs all the real work"""
        self.refreshLayerLists()
        self.dlg.lineLayerChooser.currentIndexChanged.connect(self.updateFieldExpressionWidgets)
        self.dlg.nodeLayerEnabledBox.stateChanged.connect(self.nodeLayerEnabledStateChanged)
        # Add line layers to the combo box
        self.dlg.lineLayerChooser.clear()
        self.dlg.lineLayerChooser.addItems(self.lineLayerList)
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            selectedLayer = self.layers[self.lineLayerList.currentIndex]  # type: QgsVectorLayer
            # Get the Node Threshold
            nodeThreshold = self.dlg.nodeThresholdBox.value()
            nIter = self.dlg.iterationsBox.value()
            try:
                JennyAlgorithm.run(iface=self.iface, snapThreshold=nodeThreshold, lineLayer=selectedLayer,
                                   iterations=nIter)
            except Exception as exception:
                self.iface.messageBar().pushMessage("Flow Map Arrow Curver", "Operation Failed! An Error Occurred.",
                                                    level=QgsMessageBar.WARNING, duration=5)
                raise exception
            else:
                self.iface.messageBar().pushMessage("Flow Map Arrow Curver", "Operation Complete",
                                                    level=QgsMessageBar.INFO, duration=3)
