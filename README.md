# Flow Map Curved Arrow Calculator - QGIS Plugin

This is a QGIS plugin which can be used to generate curved flow lines for use in creating an OD flow map, such as the
one seen below:

![Example Flow Map](images/flowmapexample.png)

Source: Jenny et al (2016)


The plugin uses iterative force-based approach in which each flow is represented by a bezier curve. Each curve's
control point has various forces acting on it which cause it to be repelled from other nodes and curves, attracted 
to the straight-line midpoint, and to improve angular resolution at nodes. The algorithm is based on that described in
the paper [Force-directed layout of origin-destination flow maps](http://dx.doi.org/10.1080/13658816.2017.1307378)
by Jenny, et al.

## Getting Started

### Prerequisites

QGIS >=2.0

The plugin has been tested on QGIS 2.18 on Linux.

### Installing

Copy files into a folder in your QGIS plugins directory and enable the plugin in the QGIS plugin manager.

## Contributing

Pull requests and bug reports are welcome.

## Authors

* **Patrick Malcolm** - *Initial work* - [patmalcolm91](https://github.com/patmalcolm91)
