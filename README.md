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

After installing you can start the two examples by executing `grgen_example_naca` or `grgen_example_sphere` in the terminal.
If all necessary packages are installed, you can just execute the example file in the source folder with `python3 nacaExample.py` or `python3 sphereExample.py` without installation.

Since this project is in a pre-alpha stage and just a proof of concept there is currently no way to output the grid files. If you want to use different input geometries follow the two examples for a [NACA profile](src/grgen/nacaExample.py) and a [sphere](src/grgen/sphereExample.py). 

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
