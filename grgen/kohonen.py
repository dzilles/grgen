# Copyright (c) 2020 Daniel Zilles
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import scipy.spatial
import random
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from grgen.auxiliary import Timer
from grgen.auxiliary import plot
import matplotlib.pyplot as plt

"""
Implementation of the Kohonen self-organizing map where a grid is trained to represent some input geometry.
"""

class Kohonen:

    """ The class of the self-organizing map """

    def __init__(self, spacing, geometry, dim=2, s=0.1, iterations=None, iterationsFactor=1, minRadius=None, maxRadius=None, training="online", batchSize = 25, gridType="unstructured", vertexType="triangular"):

        """ Initialization of the Kohonen class 

            :param spacing: approximation of the grid spacing used to build the initial grid
            :param geometry: geometry as sets of vertices inside a list. First entry is the outer boundary.
            :param dim: dimensions 2 or 3, TODO implement 1D, 3D
            :param s: constant for the lateral connection of two neurons
            :param iterations: maximum number of iterations
            :param iterationsFactor: Factor to increase/decrease default iteration number
            :param minRadius: minimum radius
            :param maxRadius: maximum radius
            :param training: "batch", "online" TODO implement batch
            :param batchSize: size of the training data for mini-batch learning
            :param gridType: "structured", "unstructured" TODO implement structured
            :param vertexType: "triangular", "rectangular" TODO implement rectangular
        """

        self.spacing = spacing
        self.geometry = geometry
        self.dim = dim
        self.s = s
        self.iterations = iterations
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.training = training
        self.batchSize = batchSize
        self.gridType = gridType
        self.vertexType = vertexType

        # Weights of the Kohonen network. Also the grid coordinates of all cells.
        self.weights = None
        self.startWeights = None
        self.noCells = None
        self.noPoints = None
        self.noInternalPoints = None
        self.noBoundaryPoints = None
        # Minimum and maximum coordinates of the geometry
        self.boundingBox = None

        self.eps = 10e-5

        # Storage for the learning operation
        self.randomInput = None
        self.squaredDistance = None
        self.manDistance = None
        self.lateralConnection = None
        self.geometryProbability = None
        self.vertexProbability = None

        # Fixed topology of the grid
        self.connection = None
        self.neighbors = None
        self.boundary = None
        self.boundaryPoints = None
        self.innerPoints = None
        self.boundaryId = None
        self.boundaryFace = None

        # auxiliary
        self.timer = Timer()

        # Initialize som algorithm

        # 1) Calculate bounding box
        self.getBoundingBox()
        if maxRadius == None:
            delta = np.subtract(self.boundingBox[:,1], self.boundingBox[:,0])
            self.maxRadius = np.max(delta)/spacing + 10
        if minRadius == None:
            self.minRadius = 2

        # 2) Initialize weights of the network
        self.buildWeights()

        # 3) Remove coordinates inside inner geometry or outside outer boundary
        self.removeGridCoordinates()

        # 3) Build the grid topology (connections, cell neighbors, ...)
        self.buildGridTopology()

        if iterations == None:
            self.iterations = 10*self.noPoints
        self.iterations = int(iterationsFactor*self.iterations)

        self.calculateBoundaryProbability()

    def getBoundingBox(self):
        """ Calculate the bounding box of the input geometry """

        self.timer.startTimer("getBoundingBox")
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

        self.boundingBox = np.concatenate((a, b), axis = 1)
        self.timer.stopTimer("getBoundingBox")

    def buildWeights(self):
        """ Calculate weights (the initial coordinates of the grid) """

        self.timer.startTimer("buildWeights")

        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        if(self.vertexType == "triangular"):
            spacingY = np.sqrt(self.spacing**2 - (self.spacing/2)**2)
        else:
            spacingY = self.spacing

        rangeX = np.arange(minX-3*self.spacing, maxX+3*self.spacing, self.spacing)
        rangeY = np.arange(minY-3*spacingY, maxY+3*spacingY, spacingY)

        x, y = np.meshgrid(rangeX, rangeY)

        if(self.vertexType == "triangular"):
            x[::2,:]+=self.spacing/2

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        self.weights = np.concatenate((x, y), axis = 1)
        self.noPoints = np.shape(self.weights)[0]

        self.timer.stopTimer("buildWeights")

    def removeGridCoordinates(self):
        """ Remove coordinates inside geometry, TODO: extreme slow for large grids"""

        self.timer.startTimer("removeGridCoordinates")
        removeCoord = np.ones((tf.shape(self.weights)[0]), dtype=bool)

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        for i in range(0, np.shape(self.weights)[0]):

            point = Point(self.weights[i,0], self.weights[i,1])

            if(inner.contains(point)):
                removeCoord[i] = False
            else:
                if(outer.contains(point)):
                    removeCoord[i] = True
                else:
                    removeCoord[i] = False

        self.weights = self.weights[removeCoord,:]
        self.startWeights = self.weights
        self.noPoints = np.shape(self.weights)[0]

        self.timer.stopTimer("removeGridCoordinates")

    def buildGridTopology(self):
        """ Grid topology, TODO: extreme slow, add rectangular grid """

        self.timer.startTimer("buildGridTopology")

        triangulation = scipy.spatial.Delaunay(self.weights)
        self.connection = triangulation.simplices
        self.neighbors = triangulation.neighbors
    
        it = 0
        remove = list()

        for x in self.connection:

            vertex = tf.gather(self.weights, x, axis=0)
            minimum = tf.math.reduce_min(vertex, axis=0)
            maximum = tf.math.reduce_max(vertex, axis=0)

            if((maximum[0]-minimum[0])*(maximum[1]-minimum[1])/2 > self.spacing**2/2+self.eps):
                remove.append(it)

            it+=1

        self.connection =np.delete(self.connection, remove, axis=0)

        self.neighbors[np.isin(self.neighbors, remove)] = -1
        self.neighbors = np.delete(self.neighbors, remove, axis=0)

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
        self.innerPoints = np.arange(0, self.noPoints, 1, dtype=np.int32)
        self.innerPoints = np.delete(self.innerPoints, self.boundaryPoints)

        self.noCells = np.shape(self.connection)[0]

        self.noInternalPoints = np.shape(self.boundaryPoints)[0]
        self.noBoundaryPoints = np.shape(self.innerPoints)[0]

        self.timer.stopTimer("buildGridTopology")

    def produceRandomInput(self, tensorflow=True):
        """ produce random point for the learning step """

        self.timer.startTimer()

        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        while(True):

            randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)])

            point = Point(randomCoordinate[0], randomCoordinate[1])

            if(inner.contains(point)):
                continue
            else:
                if(outer.contains(point)):
                    if (tensorflow):
                        return tf.Variable(randomCoordinate, dtype=np.float32)
                    else:
                        return randomCoordinate
                else:
                    continue
        self.timer.stopTimer("produceRandomInput")

    def calculateBoundaryProbability(self):
        """ helper function for produceRandomInputBoundary() """

        self.geometryProbability = list()
        self.vertexProbability = list()

        for idx in range(0, len(self.geometry)):

            self.vertexProbability.append( np.sqrt(np.sum((self.geometry[idx] - np.roll(self.geometry[idx], 1, axis=0))**2, axis=1)) )

            self.geometryProbability.append( np.sum(self.vertexProbability[idx], axis=0) )

            self.vertexProbability[idx] = self.vertexProbability[idx]/np.sum(self.vertexProbability[idx])

        self.geometryProbability = self.geometryProbability/np.sum(self.geometryProbability)

    def produceRandomInputBoundary(self, tensorflow=True):
        """ produce random point for the learning step on the boundary """

        self.timer.startTimer("produceRandomInputBoundary")

        idx = np.random.choice(len(self.geometry), size = 1, p=self.geometryProbability )

        idx=int(idx)

        nbr = np.shape(self.geometry[idx])[0]

        v = np.random.choice(nbr, size = 1, p=self.vertexProbability[idx])

        minX = self.geometry[idx][v,0]
        minY = self.geometry[idx][v,1]
        maxX = np.roll(self.geometry[idx], 1, axis=0)[v,0]
        maxY = np.roll(self.geometry[idx], 1, axis=0)[v,1]

        randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)]).reshape(-1,)

        self.timer.stopTimer("produceRandomInputBoundary")

        if (tensorflow):
            return tf.Variable(randomCoordinate, dtype=np.float32)
        else:
            return randomCoordinate

    def produceRandomBatch(self):
        """ produce a batch of random points """

        batchData = np.zeros((self.batchSize, self.dim))

        for i in range(0, self.batchSize):

            batchData[i,:] = self.produceRandomInput(False)

        return tf.Variable(batchData, dtype=np.float32)

    def moveBoundaryPoints(self):
        """ move boundary weights/points on the geometry boundary """

        self.timer.startTimer("moveBoundaryPoints")

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])
        
        movement = np.zeros((self.noBoundaryPoints,2))

        for idx in range(0, tf.shape(self.weightsBoundary)[0]):        

            point = Point(self.weightsBoundary[idx,0], self.weightsBoundary[idx,1])
    
            pOuter, p = nearest_points(outer.boundary, point)
            pInner, p = nearest_points(inner.boundary, point)
            
            if(point.distance(pInner) > point.distance(pOuter)):
            
                movement[idx,0] = pOuter.x
                movement[idx,1] = pOuter.y
            else:
                movement[idx,0] = pInner.x
                movement[idx,1] = pInner.y

        self.weightsBoundary = movement
        self.weights[self.boundaryPoints,:] = movement

        self.timer.stopTimer("moveBoundaryPoints")

    def trainingOperation(self, it, inputData, searchSet, searchSetStart, trainingSetStart):
        """ ordering stage for all cells """

        squaredDistance = tf.reduce_sum( (searchSet - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(squaredDistance, axis=0)

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = tf.cast((it)**(-0.2) * X, dtype=np.float32)

        squaredDistanceStart = tf.cast(tf.reduce_sum( (trainingSetStart - tf.expand_dims(searchSetStart[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = tf.cast(self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25)),dtype=np.float32)

        lateralConnection = self.s**(squaredDistanceStart/(radius**2))

        self.tmpWeights = tf.Variable(self.tmpWeights + (tf.expand_dims(delta*lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.tmpWeights)), dtype=np.float32)

    def smoothingOperation(self, it, inputData, searchSet, searchSetStart, trainingSetStart):
        """ ordering stage for all cells """

        squaredDistance = tf.reduce_sum( (searchSet - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(squaredDistance, axis=0)

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = tf.cast((it)**(-0.2) * X, dtype=np.float32)

        squaredDistanceStart = tf.cast(tf.reduce_sum( (trainingSetStart - tf.expand_dims(searchSetStart[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = tf.cast(self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25)),dtype=np.float32)

        lateralConnection = self.s**(squaredDistanceStart/(radius**2))

        self.tmpWeights = tf.Variable(self.tmpWeights + (tf.expand_dims(delta*lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.tmpWeights)), dtype=np.float32)

    def train(self):
        """ train the grid """

        self.timer.startTimer("train")

        self.weights = tf.Variable(self.weights, dtype=np.float32)
        self.tmpWeights = tf.gather(self.weights, self.boundaryPoints)
        searchSet = tf.cast(tf.gather(self.weights, self.boundaryPoints), dtype=np.float32)
        searchSetStart = tf.gather(self.startWeights, self.boundaryPoints)
        trainingSetStart = tf.gather(self.startWeights, self.boundaryPoints)

        for it in range(1, int(self.iterations)):
    
            self.trainingOperation(it, 
                                   self.produceRandomInputBoundary(), 
                                   searchSet,
                                   searchSetStart,
                                   trainingSetStart)

            if(it%200==0):

        tf.compat.v1.scatter_update(self.weights, self.boundaryPoints, self.tmpWeights)
        self.timer.stopTimer("train")

    def smooth(self):
        """ smooth the grid """

        self.timer.startTimer("smoothing")

        alpha_prob = self.noInternalPoints/(self.noBoundaryPoints*4 + self.noInternalPoints)

        for it in range(1, int(self.iterations)):

            alpha = np.random.uniform(0, 1, 1)

            #self.smoothingInternal(it)
            #if(alpha > alpha_prob):
            #    self.smoothingBoundary(it)
            #else:
            #    self.smoothingInternal(it)

        self.timer.stopTimer("smoothing")

    def summary(self):
        """ Print a few grid information """

        print("_________________________________________________________")
        print("                        ")
        print("Summary of the grid")
        print("_________________________________________________________")
        print("spacing:         ", self.spacing)
        print("dimension:       ", self.dim)
        print("minimum x:       ", self.boundingBox[0,0])
        print("maximum x:       ", self.boundingBox[0,1])
        print("minimum y:       ", self.boundingBox[1,0])
        print("maximum y:       ", self.boundingBox[1,1])
        print("s:               ", self.s)
        print("iterations:      ", self.iterations)
        print("minRadius :      ", self.minRadius)
        print("maxRadius:       ", self.maxRadius)
        print("noPoints         ", self.noPoints)
        print("noCells:         ", np.shape(self.connection)[0])
        print("noBoundaryCells: ", np.shape(self.boundary)[0])
        print("_________________________________________________________")
