from grgen import Kohonen
import numpy as np

def buildPolygonNACA(xx):

    t = xx/100

    x = np.linspace(0.0, 1.0, 200)

    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    x = np.concatenate((x, np.flip(x)), axis = 0).reshape(-1,1)
    yt = np.concatenate((yt, np.flip(-yt)), axis = 0).reshape(-1,1)
    naca = np.concatenate((x, yt), axis = 1)

    outer = np.array(np.mat('-1 0.5; 2 0.5; 2 -0.5; -1 -0.5; -1 0.5'))

    geometry = list()
    geometry.append(outer)
    geometry.append(naca)

    return geometry

def main():

    geometry = buildPolygonNACA(30)
    som = Kohonen(0.05, geometry, training = "batch")
    som.summary()
    som.train()
    som.smoothing()
    som.printTimerSummary()

if __name__ == "__main__":

    main()


