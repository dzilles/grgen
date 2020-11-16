import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
import scipy.spatial
import random
import sys
import time
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


class Kohonen:

    def __init__(self, spacing, geometry, dim=2, s=0.1, iterations=None, minRadius=1, maxRadius=None):

        self.startTime = None
        self.timerSummary = dict()

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
        self.weightsBoundary = None
        self.weightsInternal = None
        self.size = tf.shape(self.weights)[0]
        self.removeGridConnection()


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


        self.mask = None

        # Storage for the learning operation
        self.randomInput = self.produceRandomInput
        self.squaredDistance = None
        self.manDistance = None
        self.lateralConnection = None

        # Fixed topology of the grid
        self.connection = None
        self.neighbors = None
        self.boundary = None
        self.boundaryPoints = None
        self.innerPoints = None
        self.boundaryId = None
        self.boundaryFace = None
        self.eps = 10e-5
    
        self.buildGridTopology()

        self.weightsBoundary = self.weights[self.boundaryPoints,:]

        self.startWeights = self.weights

        self.moveBoundaryPoints()

        self.weights = tf.Variable(self.weights, dtype=np.float32)
        self.startWeights = tf.Variable(self.startWeights, dtype=np.float32)

        self.weightsInner = tf.gather(self.weights, self.innerPoints, axis=0)
        self.startWeightsInner = tf.gather(self.startWeights, self.innerPoints, axis=0)
        self.startWeightsBoundary = tf.gather(self.startWeights, self.boundaryPoints, axis=0)
        self.weightsBoundary = tf.gather(self.weights, self.boundaryPoints, axis=0)

        self.noInternalCells = np.shape(self.weightsInner)[0]
        self.noBoundaryCells = np.shape(self.weightsBoundary)[0]
        
        # Grid quality

        self.cellAspectRatio = None
        self.skewness = None
        self.orthogonality = None
        self.smoothness = None

        self.calculateGridQuality()


    def startTimer(self):

        self.startTime = time.time()

    def stopTimer(self, functionName):

        if functionName in self.timerSummary:

            self.timerSummary[functionName] += time.time() - self.startTime
        else:
            self.timerSummary[functionName] = time.time() - self.startTime
        self.printTimerSummary()


    def printTimerSummary(self):

        for x in self.timerSummary:
            print(x, ": ", self.timerSummary[x])


    def calculateGridQuality(self):

        for c in self.connection:
        
            print("Platzhalter")

            


    def buildGridTopology(self):

        self.startTimer()

        triangulation = scipy.spatial.Delaunay(self.weights)
        self.connection = triangulation.simplices
        self.neighbors = triangulation.neighbors
        self.stopTimer("Triangulation")
    
        it = 0
        remove = list()

        self.startTimer()
        for x in self.connection:

            vertex = tf.gather(self.weights, x, axis=0)
            minimum = tf.math.reduce_min(vertex, axis=0)
            maximum = tf.math.reduce_max(vertex, axis=0)

            if((maximum[0]-minimum[0])*(maximum[1]-minimum[1])/2 > self.spacing**2/2+self.eps):
                remove.append(it)

            it+=1

        self.connection =np.delete(self.connection, remove, axis=0)
        self.startTimer()
        for r in remove:
            self.neighbors = np.where(self.neighbors == r, -1, self.neighbors)

        self.stopTimer("removeConnection")
        self.neighbors = np.delete(self.neighbors, remove, axis=0)


        self.startTimer()
        self.boundary = np.argwhere(self.neighbors < 0)

        tmpBndry = list()
        for b in self.boundary:

            if (b[1]==0):
                tmpBndry.append(self.connection[b[0],1])
                tmpBndry.append(self.connection[b[0],2])
            if (b[1]==1):
                tmpBndry.append(self.connection[b[0],2])
                tmpBndry.append(self.connection[b[0],0])
            if (b[1]==2):
                tmpBndry.append(self.connection[b[0],0])
                tmpBndry.append(self.connection[b[0],1])
    
        self.boundaryPoints = np.unique(np.array(tmpBndry))
        self.innerPoints = np.arange(0, self.size, 1, dtype=np.int32)
        self.innerPoints = np.delete(self.innerPoints, self.boundaryPoints)
        
        self.stopTimer("rest")

    def moveBoundaryPoints(self):

        self.startTimer()

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        for idx in range(0, tf.shape(self.weightsBoundary)[0]):        

            point = Point(self.weightsBoundary[idx,0], self.weightsBoundary[idx,1])
    
            #pInner, _ = nearest_points(inner, point)
            pOuter, p = nearest_points(outer.boundary, point)
            pInner, p = nearest_points(inner.boundary, point)
            
            if(point.distance(pInner) > point.distance(pOuter)):
            
                self.weightsBoundary[idx,0] = pOuter.x
                self.weightsBoundary[idx,1] = pOuter.y
                self.weights[self.boundaryPoints[idx],0] = pOuter.x
                self.weights[self.boundaryPoints[idx],1] = pOuter.y
            else:
                self.weightsBoundary[idx,0] = pInner.x
                self.weightsBoundary[idx,1] = pInner.y
                self.weights[self.boundaryPoints[idx],0] = pInner.x
                self.weights[self.boundaryPoints[idx],1] = pInner.y


        self.stopTimer("moveBoundaryPoints")

    def summary(self):

        print("Summary of the grid")
        print("________________________")
        print("spacing:         ", self.spacing)
        print("dimension:       ", self.dim)
        print("minimum x:       ", self.boundingBox[0,0])
        print("maximum x:       ", self.boundingBox[0,1])
        print("minimum y:       ", self.boundingBox[1,0])
        print("maximum y:       ", self.boundingBox[1,1])
        print("s:               ", self.s)
        print("size:            ", self.size)
        print("iterations:      ", self.iterations)
        print("minRadius :      ", self.minRadius)
        print("maxRadius:       ", self.maxRadius)
        print("noCells:         ", np.shape(self.connection)[0])
        print("noBoundaryCells: ", np.shape(self.boundary)[0])
        print("________________________")


    # Return the minimum and maximum expansion of the grid
    # boundingBox[x=0, y=1, z=2, min=0/max=1]

    def getBoundingBox(self):

        self.startTimer()
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

        self.stopTimer("getBoundingBox")
        return np.concatenate((a, b), axis = 1)

    def buildWeights(self):

        self.startTimer()
        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        spacingY = np.sqrt(self.spacing**2 - (self.spacing/2)**2)

        rangeX = np.arange(minX-3*self.spacing, maxX+3*self.spacing, self.spacing)
        rangeY = np.arange(minY-3*spacingY, maxY+3*spacingY, spacingY)

        x, y = np.meshgrid(rangeX, rangeY)

        x[::2,:]+=self.spacing/2

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        self.stopTimer("buildWeights")
        return np.concatenate((x, y), axis = 1)

    def removeGridConnection(self):

        self.startTimer()
        removeCoord = np.ones((self.size), dtype=bool)

        outer = mpltPath.Path(self.geometry[0])
        inner = mpltPath.Path(self.geometry[1])

        for i in range(0, self.size):

            print(i/self.size*100)

            coord = self.weights[i,:]

            if(inner.contains_point(coord)):
                removeCoord[i] = False
            else:
                if(outer.contains_point(coord)):
                    removeCoord[i] = True
                else:
                    removeCoord[i] = False

        self.weights = self.weights[removeCoord,:]
        self.size = np.shape(self.weights)[0]

        self.stopTimer("removeGridConnection")

    def produceRandomInput(self):

        self.startTimer()

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
        self.stopTimer("produceRandomInput")

    def trainingOperationGeneral(self,it):

        inputData = self.produceRandomInput()
        self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weightsInner - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        if(np.any(self.boundaryPoints[:] == bmuIndex) ):
            inputData = self.weights[bmuIndex,:]

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = (it)**(-0.2) * X

        self.manDistance = tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeightsInner[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25))

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        #self.weights = self.weights + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weights))

        #self.lateralConnection = tf.gather(self.lateralConnection, self.innerPoints)

        self.weightsInner = self.weightsInner + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInner))

        tf.compat.v1.scatter_update(self.weights, self.innerPoints, self.weightsInner)

    def trainingOperationInternal(self,it):

        inputData = self.produceRandomInput()
        self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weightsInner - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        #if(np.any(self.boundaryPoints[:] == bmuIndex) ):
        #    inputData = self.weights[bmuIndex,:]

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)
        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = (it)**(-0.2) * X

        self.manDistance = tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeightsInner[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25))

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        #self.lateralConnection = tf.gather(self.lateralConnection, self.innerPoints)

        self.weightsInner = self.weightsInner + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInner))

        tf.compat.v1.scatter_update(self.weights, self.innerPoints, self.weightsInner)

    def trainingOperationBoundary(self,it):
        print("Platzhalter")

    def trainingOperationSmoothingInternal(self,it):

        inputData = self.produceRandomInput()
        self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weights - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        if(np.any(self.boundaryPoints[:] == bmuIndex) ):
            inputData = self.weights[bmuIndex,:]

        delta = 0.03

        self.manDistance = tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = 4

        self.s = 0.05

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        #self.lateralConnection = tf.gather(self.lateralConnection, self.innerPoints)

        self.weightsInner = self.weightsInner + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInner))

        tf.compat.v1.scatter_update(self.weights, self.innerPoints, self.weightsInner)

    def trainingOperationSmoothingBoundary(self,it):

        randInt = np.random.randint(0, tf.shape(self.weightsBoundary)[0])
        inputData = self.weightsBoundary[randInt, :]
        #inputData = self.produceRandomInput()
        #self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weightsBoundary - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        bmuIndex = self.boundaryPoints[bmuIndex]

        delta = 0.03

        self.manDistance = tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = 4

        self.s = 0.05

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weightsInner = self.weightsInner + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInner))

        tf.compat.v1.scatter_update(self.weights, self.innerPoints, self.weightsInner)

    def trainingOperationSmoothingBoundary2(self,it):

        randInt = np.random.randint(0, tf.shape(self.weightsBoundary)[0])
        inputData = self.weightsBoundary[randInt, :]
        #inputData = self.produceRandomInput()
        #self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weightsBoundary - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        delta = 0.03

        self.manDistance = tf.reduce_sum( (self.startWeightsBoundary - tf.expand_dims(self.startWeightsBoundary[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = 4

        self.s = 0.05

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weightsBoundary = self.weightsBoundary + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsBoundary))

        tf.compat.v1.scatter_update(self.weights, self.boundaryPoints, self.weightsBoundary)




    def train(self):

        self.startTimer()

        print("Ordering stage")
        #for it in range(1, int(self.iterations/10)):


         #   if(it%100 == 0):
          #      plt.clf()
           #     plt.triplot(self.weights[:,0], self.weights[:,1], self.connection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1], c = self.lateralConnection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1]) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
            #    plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
             #   plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
              #  plt.draw()
               # plt.pause(0.0001)
            #self.trainingOperationGeneral(it)

      #  print("Refine stage")
       # for it in range(int(0.1*self.iterations), self.iterations + 1):

#            print(it)

        #    if(it%100 == 0):
         #       plt.clf()
          #      plt.triplot(self.weights[:,0], self.weights[:,1], self.connection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1], c = self.lateralConnection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1]) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
           #     plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
            #    plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
             #   plt.draw()
              #  plt.pause(0.0001)
            #self.trainingOperationInternal(it)
        print("Smoothing stage")
        for it in range(1, int(self.iterations)):

#            print(it)

            if(it%100 == 0):
                plt.clf()
                plt.triplot(self.weights[:,0], self.weights[:,1], self.connection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1], c = self.lateralConnection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1]) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
                #plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
                #plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.draw()
                plt.pause(0.0001)

            alpha_prob = self.noInternalCells/(self.noBoundaryCells*4 + self.noInternalCells)

            alpha = np.random.uniform(0, 1, 1)

            if(alpha > alpha_prob):
                self.trainingOperationSmoothingBoundary(it)
            else:
                self.trainingOperationSmoothingInternal(it)
        plt.clf()
        plt.triplot(self.weights[:,0], self.weights[:,1], self.connection) 
        #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1], c = self.lateralConnection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1]) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
        plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
        plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
        plt.show()

        self.stopTimer("train")
 

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

    geometry = buildPolygonNACA(30)
    som = Kohonen(0.05, geometry)

    som.summary()

    som.printTimerSummary()

    som.train()

    som.printTimerSummary()

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
        


