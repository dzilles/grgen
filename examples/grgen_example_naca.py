import numpy as np
# 1) After installation, the package can be imported as follows
from grgen.kohonen import Kohonen

def buildPolygonNACA(xx):

    t = xx/100

    x = np.linspace(0.0, 1.0, 200)

    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    x = np.concatenate((x, np.flip(x)), axis = 0).reshape(-1,1)
    yt = np.concatenate((yt, np.flip(-yt)), axis = 0).reshape(-1,1)
    naca = np.concatenate((x, yt), axis = 1)

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
    geometry = buildPolygonNACA(30)

    # 3) The model is initialized
    som = Kohonen(0.05, geometry, training = "online")

    # 4) The training can be started after the initialization of the model
    som.train()

    # 5) A different algorithm is used for the smoothing of the grid
    som.smoothing()

if __name__ == '__main__':
    main()
