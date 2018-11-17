# Flow Map Curved Arrow Calculator - QGIS Plugin

This is an attempt to create a QGIS plugin which can be used to generate curved flow arrows for use in creating an OD
flow map.

The intent is to use an iterative force-based approach in which the middle point of the arc ("control point") is
repelled from nodes, and attracted to the straight-line midpoint. The algorithm will be loosely based on that described
in the paper [Automated layout of
origin–destination flow maps: U.S. county-to-county migration 2009
–2013](https://doi.org/10.1080/17445647.2017.1313788).

## Getting Started

### Prerequisites

QGIS >=2.0

### Installing

Copy files into a folder in your QGIS plugins directory.

## Contributing

Pull requests and bug reports welcome.

## Authors

* **Patrick Malcolm** - *Initial work* - [patmalcolm91](https://github.com/patmalcolm91)
