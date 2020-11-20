# Mesh generation using Kohonen self-organizing maps

A mesh generator using machine learning. In particular the software uses the Kohonen self organizing map
based on the work of T. Kohonen [[1]](#1) and O. Nechaeva [[2]](#2) where some input structured or unstructured mesh is trained to represent the input geometry. The following animation shows the learning process for a NACA geometry:
![NACA](https://github.com/dzilles/grgen/raw/main/examples/output/naca.gif)
Currently this project is in a **pre-alpha stage and has a few limitations**:

- limited speed especially for large meshes: it may be possible to address this problem by using mini-batch learning for self-organizing maps
- only unstructured triangular meshes: it is possible to use this algorithm for structured meshes since the mesh topology remains
- adaptive meshes are not implemented
- grid quality can't be guaranteed
- only 2-dimensionsal

----

## Installation

The software is installed by executing the command 'pip install -e .' in the main folder.
The source for this project is available [here](src/grgen).

----

## Usage

After installing, the two examples in the [examples folder](examples) can be started by executing 
`python grgen_example_naca.py` or `python grgen_example_sphere.py`.
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
