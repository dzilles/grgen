import numpy as np
# 1) After installation, the package can be imported as follows
from grgen.kohonen import Kohonen

def buildPolygonSphere(r):

    phi = np.linspace(0.0, 2*np.pi, 200)

    x = r*np.cos(phi).reshape(-1,1)
    y = r*np.sin(phi).reshape(-1,1)

    naca = np.concatenate((x, y), axis = 1)

    outer = np.array(np.mat('-1 0.5; 2 0.5; 2 -0.5; -1 -0.5; -1 0.5'))

    # 2.1) The geometry is a list object containing the vertex geometries
    geometry = list()

    # 2.2) The first entry must contain the outer boundary
    geometry.append(outer)

    # 2.3) The following entries can contain the inner geometries
    geometry.append(naca)

    return geometry

def main():

    # 2) First the geometry is built
    geometry = buildPolygonSphere(0.3)

    # 3) The model is initialized
    som = Kohonen(0.02, geometry, training = "online")

    # 4) The training can be started after the initialization of the model
    som.training()

    # 5) A different algorithm is used for the smoothing of the grid
    som.smoothing()

if __name__ == '__main__':
    main()
