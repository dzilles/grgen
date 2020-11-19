# Mesh generation using Kohonen self-organizing maps

A mesh generator using machine learning. In particular the software uses the Kohonen self organizing map
based on the work of T. Kohonen [[1]](#1) and O. Nechaeva [[2]](#2) where some input structured or unstructured mesh is trained to represent the input geometry. The following animation shows the learning process for a NACA geometry:
![NACA](https://github.com/dzilles/grgen/raw/main/data/naca.gif)

----

## Installation

The software is installed by executing the command 'pip install -e .' in the main folder.
The source for this project is available [here](src/grgen).

----

## Usage

After installing, the two examples in the [examples folder](examples) can be started by executing 
`python grgen_example_naca` or `python grgen_example_sphere`.
The exact usage of the package is documented in these example files.

----

## References
<a id="1">[1]</a> 
Kohonen, T. (2012). 
Self-organizing  maps.
Communications of the ACM, 11(3), 147-148.
Springer Science & Business Media V.30, Springer.


<a id="2">[2]</a> 
Nechaeva, O. (2006). 
Composite algorithm for adaptive mesh construction based on self-organizing maps. 
International Conference on Artificial Neural Networks, Springer, Berlin, Heidelberg, 147-148.
