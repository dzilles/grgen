import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
import random
#import tensorflow_transform as tft
#import tensorflow_probability as tfp

class Kohonen:

    def __init__(self, spacing, geometry, dim=2, s=0.1, iterations=None, minRadius=1, maxRadius=None):

        # Approximation of the grid spacing
        self.spacing = spacing

        # Geometry as sets of vertices inside a list
        self.geometry = geometry

        # Dimension of the grid, e.g. 1D/2D/3D
        self.dim = dim

        self.boundingBox = self.getBoundingBox()

        # Constant for the lateral connection of two neurons
        self.s = tf.constant(s, dtype = np.float32)

        # Weights of the neurons, also the coordinates of the output grid
        self.weights = self.buildWeights()
        self.size = tf.shape(self.weights)[0]
        self.removeGridConnection()
        self.size = tf.shape(self.weights)[0]


        if maxRadius == None:

            delta = np.subtract(self.boundingBox[:,1], self.boundingBox[:,0])

            self.maxRadius = (np.max(delta))/spacing +10
            self.maxRadius = tf.constant(self.maxRadius, dtype=np.float32)
        else:
            self.maxRadius = tf.constant(maxRadius, dtype=np.float32)

        self.minRadius = tf.constant(minRadius, dtype=np.float32)
        if iterations == None:
            self.iterations = 10*self.size
        else:
            self.iterations = iterations

        self.startWeights = self.weights

        self.mask = None

        # Storage for the learning operation
        self.randomInput = self.produceRandomInput
        self.squaredDistance = None
        self.manDistance = None
        self.lateralConnection = None

    def summary(self):

        print("Summary of the grid: ")
        print("spacing:    ", self.spacing)
        print("dimension:  ", self.dim)
        print("minimum x:  ", self.boundingBox[0,0])
        print("maximum x:  ", self.boundingBox[0,1])
        print("minimum y:  ", self.boundingBox[1,0])
        print("maximum y:  ", self.boundingBox[1,1])
        print("s:          ", self.s)
        print("size:       ", self.size)
        print("iterations: ", self.iterations)
        print("minRadius : ", self.minRadius)
        print("maxRadius : ", self.maxRadius)

    # Return the minimum and maximum expansion of the grid
    # boundingBox[x=0, y=1, z=2, min=0/max=1]

    def getBoundingBox(self):

        boundingBox = np.zeros((self.dim, 2, len(self.geometry)))
        index = 0

        for g in self.geometry:

            boundingBox[0, 0, index] = np.min(g[:,0])
            boundingBox[0, 1, index] = np.max(g[:,0])
            boundingBox[1, 0, index] = np.min(g[:,1])
            boundingBox[1, 1, index] = np.max(g[:,1])
            index += 1

        a = np.min(boundingBox[:,0,:], axis =1).reshape(-1,1)
        b = np.max(boundingBox[:,1,:], axis =1).reshape(-1,1)

        return np.concatenate((a, b), axis = 1)

    def buildWeights(self):

        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        rangeX = np.arange(minX-3*self.spacing, maxX+3*self.spacing, self.spacing)
        rangeY = np.arange(minY-3*self.spacing, maxY+3*self.spacing, self.spacing)

        x, y = np.meshgrid(rangeX, rangeY)

        self.connectGrid(np.size(rangeX), np.size(rangeY))

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        return tf.Variable(np.concatenate((x, y), axis = 1), dtype=np.float32)

    def connectGrid(self, sizeX, sizeY):

        print("Platzhalter")

        #self.connection = np.zeros((sizeX*sizeY, 3))

        #index = 0

        #for i in range(0, sizeX):
        #    for j in range(0, sizeX):

        #        self.connection[index, 0] = 

    def removeGridConnection(self):

        removeCoord = np.ones((self.size), dtype=bool)

        outer = mpltPath.Path(self.geometry[0])
        inner = mpltPath.Path(self.geometry[1])

        for i in range(0, self.size):

            coord = self.weights[i,:]

            if(inner.contains_point(coord)):
                removeCoord[i] = False
            else:
                if(outer.contains_point(coord)):
                    removeCoord[i] = True
                else:
                    removeCoord[i] = False

            print(removeCoord[i])

        print(self.weights)

        self.weights = tf.boolean_mask(self.weights, removeCoord, axis=0)
        print(self.weights)

    def produceRandomInput(self):

        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        outer = mpltPath.Path(self.geometry[0])
        inner = mpltPath.Path(self.geometry[1])

        while(True):

            randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)])
            p = randomCoordinate.reshape(1,-1)

            if(inner.contains_points(p)):
                continue
            else:
                if(outer.contains_points(p)):
                    return tf.Variable(randomCoordinate, dtype=np.float32)
                else:
                    continue

    def trainingOperation(self,it):

        inputData = self.produceRandomInput()
        self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weights - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = (it)**(-0.2) * X

        self.manDistance = tf.reduce_sum( (self.startWeights - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25))

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weights = self.weights + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weights))

    def train(self):
        
        for it in range(1, self.iterations + 1):

            print(it)

            if(it%1000 == 0):
                plt.clf()
                plt.scatter(self.weights[:,0], self.weights[:,1], c=self.lateralConnection, marker='.', s=1) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
                plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
                plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
                plt.draw()
                plt.pause(0.0001)

            self.trainingOperation(it)

def buildPolygonNACA(xx):

    t = xx/100

    x = np.linspace(0.0, 1.0, 200)

    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    x = np.concatenate((x, np.flip(x)), axis = 0).reshape(-1,1)
    yt = np.concatenate((yt, np.flip(-yt)), axis = 0).reshape(-1,1)
    naca = np.concatenate((x, yt), axis = 1)

    outer = np.array(np.mat('-1 0.5; 2 0.5; 2 -0.5; -1 -0.5; -1 0.5'))

    print(outer)


    geometry = list()

    geometry.append(outer)
    geometry.append(naca)

    return geometry

def main():

    geometry = buildPolygonNACA(12)
    som = Kohonen(0.01, geometry)

    som.summary()

    som.train()

if __name__ == "__main__":

    main()







        #self.squaredDistance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.weights, axis=0),tf.expand_dims(batchData, axis=1)), 2), 2)

        #self.bmuIndices = tf.argmin(self.squaredDistance, axis=1)
        #self.bmuLocs = tf.reshape(tf.gather(self.locationVects, self.bmuIndices), [-1, 2])

        # Update the weigths 
        #radius = self.sigma - (np.float32(iter) * (self.alpha - 1)/(num_epoch - 1))

        #print(radius)

        #alpha = self.alpha - (np.float32(iter) * (self.alpha - 1)/(num_epoch - 1))

        #self.bmuSquaredDistance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.locationVects, axis=0),
        #            tf.expand_dims(self.bmuLocs, axis=1)), 2), 2)

        #self.neighbourhoodFunc = tf.exp(tf.divide(tf.negative(tf.cast(
        #        self.bmuSquaredDistance, "float32")), tf.square(radius, 1)))

        #self.learningRate = self.neighbourhoodFunc * alpha
        
        #self.numerator = tf.reduce_sum(tf.expand_dims(self.neighbourhoodFunc, axis=-1) * tf.expand_dims(batchData, axis=1), axis=0)
        #self.denominator = tf.expand_dims(
        #    tf.reduce_sum(self.neighbourhoodFunc,axis=0) + float(1e-12), axis=-1)

        #self.weights = self.numerator / self.denominator
        


