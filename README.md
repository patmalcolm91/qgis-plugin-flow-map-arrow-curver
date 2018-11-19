# Flow Map Curved Arrow Calculator - QGIS Plugin

This is an attempt to create a QGIS plugin which can be used to generate curved flow arrows for use in creating an OD
flow map.

The intent is to use an iterative force-based approach in which each flow is represented by a bezier curve. Each curve's
control point has various forces acting on it which cause it to be repelled from other nodes and curves, and attracted 
to the straight-line midpoint. The algorithm will be based on that described in the paper 
[Force-directed layout of origin-destination flow maps](http://dx.doi.org/10.1080/13658816.2017.1307378) by Jenny, et al.

## Getting Started

### Prerequisites

QGIS >=2.0

### Installing

Copy files into a folder in your QGIS plugins directory.

## Contributing

Pull requests and bug reports welcome.

## Authors

* **Patrick Malcolm** - *Initial work* - [patmalcolm91](https://github.com/patmalcolm91)
