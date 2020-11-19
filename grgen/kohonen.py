import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
import scipy.spatial
import random
import sys
import imageio
import glob, os, re
import time
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

"""
Implementation of the Kohonen self-organizing map where a grid is trained to represent some input geometry.
"""

class Kohonen:

    """ The class of the self-organizing map """

    def __init__(self, spacing, geometry, dim=2, s=0.1, iterations=None, minRadius=None, maxRadius=None, training="online", batchSize = 25, gridType="unstructured", vertexType="triangular"):

        """ The class of the self-organizing map 

            :param spacing: approximation of the grid spacing used to build the initial grid
            :param geometry: geometry as sets of vertices inside a list. First entry is the outer boundary.
            :param dim: dimensions 2 or 3 //TODO implement 1D/3D
            :param s: constant for the lateral connection of two neurons
            :param iterations: maximum number of iterations
            :param minRadius: minimum Manhatten radius
            :param maxRadius: maximum Manhatten radius
            :param training: "batch", "online" //TODO implement batch
            :param batchSize: size of the training data for mini-batch learning
            :param gridType: "structured", "unstructured" //TODO implement structured
            :param vertexType: "triangular", "rectangular" //TODO implement rectangular
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
        # Coordinates for boundary cells
        self.weightsBoundary = None
        self.startWeightsBoundary = None
        # Coordinates for internal cells
        self.weightsInternal = None
        self.startWeightsInternal = None
        self.noCells = None
        self.noPoints = None
        self.noInternalPoints = None
        self.noBoundaryPoints = None
        # Minimum and maximum coordinates of the geometry
        self.boundingBox = None

        # auxiliary data
        self.startTime = None
        self.timerSummary = dict()
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

        # Grid quality
        self.cellAspectRatio = None
        self.skewness = None
        self.orthogonality = None
        self.smoothness = None

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

        # 3)
        self.buildGridTopology()

        if iterations == None:
            self.iterations = 10*self.noPoints
        
        self.calculateGridQuality()

        self.weightsTf = tf.Variable(self.weights, dtype=np.float32)
        self.weightsInnerTf = tf.Variable(self.weightsInner, dtype=np.float32)
        self.weightsBoundaryTf = tf.Variable(self.weightsBoundary, dtype=np.float32)

        self.calculateBoundaryProbability()

    def getBoundingBox(self):
        """ Calculate the bounding box of the input geometry """

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
        self.boundingBox = np.concatenate((a, b), axis = 1)

    def buildWeights(self):
        """ Calculate weights or the initial coordinates of the grid """

        self.startTimer()
        minX = self.boundingBox[0,0]
        minY = self.boundingBox[1,0]
        maxX = self.boundingBox[0,1]
        maxY = self.boundingBox[1,1]

        if(self.vertexType == "triangular"):
            spacingY = np.sqrt(self.spacing**2 - (self.spacing/2)**2)
        else:
            spacingY = np.sqrt(self.spacing**2 - (self.spacing/2)**2)

        rangeX = np.arange(minX-3*self.spacing, maxX+3*self.spacing, self.spacing)
        rangeY = np.arange(minY-3*spacingY, maxY+3*spacingY, spacingY)

        x, y = np.meshgrid(rangeX, rangeY)

        if(self.vertexType == "triangular"):
            x[::2,:]+=self.spacing/2

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        self.stopTimer("buildWeights")
        self.weights = np.concatenate((x, y), axis = 1)
        self.noPoints = np.shape(self.weights)[0]

    def removeGridCoordinates(self):
        """ Remove coordinates inside geometry, TODO: extreme slow """

        self.startTimer()
        removeCoord = np.ones((tf.shape(self.weights)[0]), dtype=bool)

        outer = mpltPath.Path(self.geometry[0])
        inner = mpltPath.Path(self.geometry[1])

        for i in range(0, np.shape(self.weights)[0]):

            coord = self.weights[i,:]

            if(inner.contains_point(coord)):
                removeCoord[i] = False
            else:
                if(outer.contains_point(coord)):
                    removeCoord[i] = True
                #else:
                    #removeCoord[i] = False

        self.weights = self.weights[removeCoord,:]
        self.startWeights = self.weights
        self.noPoints = np.shape(self.weights)[0]

        self.stopTimer("removeGridCoordinates")

    def buildGridTopology(self):
        """ Grid topology, TODO: extreme slow, add rectangular grid """

        self.startTimer()

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
        self.startTimer()
        for r in remove:
            self.neighbors = np.where(self.neighbors == r, -1, self.neighbors)

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

        self.weightsBoundary = self.weights[self.boundaryPoints,:]
        self.weightsInner = self.weights[self.innerPoints,:]

        self.noInternalPoints = np.shape(self.weightsInner)[0]
        self.noBoundaryPoints = np.shape(self.weightsBoundary)[0]

        self.startWeightsBoundary = self.startWeights[self.boundaryPoints,:]
        self.startWeightsInner = self.startWeights[self.innerPoints,:]

        self.stopTimer("buildGridTopology")

    def calculateGridQuality(self):
        """ move boundary weights/points on the geometry boundary """

        print("Platzhalter")
        #for c in self.connection:

    def startTimer(self):
        """ move boundary weights/points on the geometry boundary """

        self.startTime = time.time()

    def stopTimer(self, functionName):
        """ move boundary weights/points on the geometry boundary """

        if functionName in self.timerSummary:

            self.timerSummary[functionName] += time.time() - self.startTime
        else:
            self.timerSummary[functionName] = time.time() - self.startTime

    def printTimerSummary(self):
        """ move boundary weights/points on the geometry boundary """

        print("_________________________________________________________")
        print("                        ")
        print("Timer")
        print("_________________________________________________________")
        for x in self.timerSummary:
            print(x, ": ", self.timerSummary[x])
        print("_________________________________________________________")

    def summary(self):
        """ move boundary weights/points on the geometry boundary """

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

    def produceRandomInput(self, tensorflow=True):
        """ move boundary weights/points on the geometry boundary """

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
                    if (tensorflow):
                        return tf.Variable(randomCoordinate, dtype=np.float32)
                    else:
                        return randomCoordinate
                else:
                    continue
        self.stopTimer("produceRandomInput")

    def calculateBoundaryProbability(self):

        self.geometryProbability = list()
        self.vertexProbability = list()

        for idx in range(0, len(self.geometry)):

            self.vertexProbability.append( np.sqrt(np.sum((self.geometry[idx] - np.roll(self.geometry[idx], 1, axis=0))**2, axis=1)) )

            self.geometryProbability.append( np.sum(self.vertexProbability[idx], axis=0) )

            self.vertexProbability[idx] = self.vertexProbability[idx]/np.sum(self.vertexProbability[idx])

        self.geometryProbability = self.geometryProbability/np.sum(self.geometryProbability)

    def produceRandomInputBoundary(self, tensorflow=True):
        """ move boundary weights/points on the geometry boundary """

        self.startTimer()

        idx = np.random.choice(len(self.geometry), size = 1, p=self.geometryProbability )

        idx=int(idx)

        nbr = np.shape(self.geometry[idx])[0]

        v = np.random.choice(nbr, size = 1, p=self.vertexProbability[idx])

        minX = self.geometry[idx][v,0]
        minY = self.geometry[idx][v,1]
        maxX = np.roll(self.geometry[idx], 1, axis=0)[v,0]
        maxY = np.roll(self.geometry[idx], 1, axis=0)[v,1]

        randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)]).reshape(-1,)

        self.stopTimer("produceRandomInputBoundary")
        if (tensorflow):
            return tf.Variable(randomCoordinate, dtype=np.float32)
        else:
            return randomCoordinate

    def produceRandomBatch(self):

        batchData = np.zeros((self.batchSize, self.dim))

        for i in range(0, self.batchSize):

            batchData[i,:] = self.produceRandomInput(False)

        return tf.Variable(batchData, dtype=np.float32)

    def trainingOperationGeneral(self,it):
        """ ordering stage for all cells """

        inputData = self.produceRandomInput()

        self.squaredDistance = tf.reduce_sum( (self.weightsTf - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = tf.cast((it)**(-0.2) * X, dtype=np.float32)

        self.manDistance = tf.cast(tf.reduce_sum( (self.startWeights - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = tf.cast(self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25)),dtype=np.float32)

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weightsTf = tf.Variable(self.weightsTf + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsTf)), dtype=np.float32)

        self.weightsBoundaryTf = tf.gather(self.weightsTf, self.boundaryPoints)
        self.weightsInnerTf = tf.gather(self.weightsTf, self.innerPoints)

    def trainingOperationGeneralBatch(self,it):
        """ ordering stage for all cells batch extension """

        batchData = self.produceRandomBatch()

        self.squaredDistance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.weightsTf, axis=0),tf.expand_dims(batchData, axis=1)), 2), 2)


        bmuIndices = tf.argmin(self.squaredDistance, axis=1)
        startW = tf.reshape(tf.gather(self.startWeights, bmuIndices), [-1, 2])

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = tf.cast((it)**(-0.2) * X, dtype=np.float32)
        iterations = tf.cast(self.iterations, dtype=np.float32)
        radius = tf.cast(self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25)),dtype=np.float32)

        self.manDistance = tf.cast(tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.startWeights, axis=0),tf.expand_dims(startW, axis=1)), 2), 2),dtype=np.float32)

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))
        
        self.numerator = tf.reduce_sum(tf.expand_dims(self.lateralConnection, axis=-1) * tf.expand_dims(batchData, axis=1), axis=0)
        self.denominator = tf.expand_dims(tf.reduce_sum(self.lateralConnection,axis=0) + float(1e-12), axis=-1)

        self.weightsTf = self.numerator / self.denominator

    def trainingOperationBoundary(self,it):
        """ refinement stage for boundary cells """

        inputData = self.produceRandomInputBoundary()

        self.squaredDistance = tf.reduce_sum( (self.weightsBoundaryTf - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = (it)**(-0.2) * X

        self.manDistance = tf.cast(tf.reduce_sum( (self.startWeightsBoundary - tf.expand_dims(self.startWeightsBoundary[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25))

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weightsBoundaryTf = self.weightsBoundaryTf + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsBoundaryTf))

        tf.compat.v1.scatter_update(self.weightsTf, self.boundaryPoints, self.weightsBoundaryTf)

    def trainingOperationInternal(self,it):
        """ refinement stage for inner cells """

        inputData = self.produceRandomInput()
        self.randomInput = inputData

        self.squaredDistance = tf.reduce_sum( (self.weightsTf - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        if(np.any(self.boundaryPoints[:] == bmuIndex) ):
            inputData = self.weightsTf[bmuIndex,:]

        X = tf.cast(1 - tf.exp(5*(it-self.iterations)/self.iterations), dtype=np.float32)
        delta = tf.cast((it)**(-0.2) * X, dtype=np.float32)

        self.manDistance = tf.cast(tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = tf.cast(self.minRadius + X*(self.maxRadius*0.05**(it/iterations) - self.minRadius)*(it**(-0.25)), dtype=np.float32)

        self.lateralConnection = delta*self.s**(self.manDistance/(radius**2))

        self.weightsInnerTf = self.weightsInnerTf + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInnerTf))

        tf.compat.v1.scatter_update(self.weightsTf, self.innerPoints, self.weightsInnerTf)

    def smoothingInternal(self,it):

        inputData = self.produceRandomInput()

        self.squaredDistance = tf.reduce_sum( (self.weightsTf - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        if(np.any(self.boundaryPoints[:] == bmuIndex) ):
            inputData = self.weightsTf[bmuIndex,:]

        delta = 0.03

        self.manDistance = tf.cast(tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeights[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing**2), dtype=np.float32)

        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = 4

        self.s = 0.05

        self.lateralConnection = tf.cast(delta*self.s**(self.manDistance/(radius**2)), dtype=np.float32)

        self.weightsInnerTf = self.weightsInnerTf + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInnerTf))

        tf.compat.v1.scatter_update(self.weightsTf, self.innerPoints, self.weightsInnerTf)

    def smoothingBoundary(self,it):

        inputData = self.produceRandomInputBoundary()

        self.squaredDistance = tf.reduce_sum( (self.weightsBoundaryTf - tf.expand_dims(inputData, axis=0))**2, axis = 1)

        bmuIndex = tf.argmin(self.squaredDistance, axis=0)

        inputData = self.weightsBoundaryTf[bmuIndex,:]
        delta = 0.03

        self.manDistance = tf.cast(tf.math.sqrt(tf.reduce_sum( (self.startWeightsInner - tf.expand_dims(self.startWeightsBoundary[bmuIndex,:], axis=0))**2, axis = 1)/(self.spacing)), dtype=np.float32)


        iterations = tf.cast(self.iterations, dtype=np.float32)

        radius = 2

        self.s = 0.05

        k = np.random.randint(1, radius)

        #self.lateralConnection = tf.cast(delta*self.s**(self.manDistance/(radius**2)), dtype=np.float32)

        self.lateralConnection = tf.cast(delta*self.s**((self.manDistance + k)**2/(radius**2))*(1 + k/self.manDistance), dtype=np.float32)


        self.weightsInnerTf = self.weightsInnerTf + (tf.expand_dims(self.lateralConnection, axis=1)*(tf.expand_dims(inputData, axis=0) - self.weightsInnerTf))

        tf.compat.v1.scatter_update(self.weightsTf, self.innerPoints, self.weightsInnerTf)

    def plot(self,it):

        fig, ax = plt.subplots() 

        plt.triplot(self.weightsTf[:,0], self.weightsTf[:,1], self.connection)
        plt.gca().set_aspect('equal', adjustable='box')

        name = 'pics/naca_' + str(it) + '.png'

        fig.savefig(name)
        plt.close(fig)

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def gif(self):

        dir_name = "pics/"
        filenames = self.sorted_alphanumeric(os.listdir(dir_name))
        print(filenames)

        with imageio.get_writer('pics/naca.gif', mode='I', fps=20) as writer:
            for filename in filenames:

                if filename.endswith(".png"):
                    image = imageio.imread(dir_name + filename)
                    writer.append_data(image)

    def removePng(self):

        dir_name = "pics/"
        filenames = os.listdir(dir_name)

        for filename in filenames:
            if filename.endswith(".png"):
                os.remove(os.path.join(dir_name, filename))

    def train(self):

        self.startTimer()

        if(self.training == "online"):

            it = 1
            print("ordering stage")
            while(it < int(0.05*self.iterations)):

                self.trainingOperationGeneral(it)
                self.draw1(it)
                it+=1
    
            print("refinement stage")

            tint_start = 1
            macroit =1

            phib = int(10*self.noBoundaryPoints/20)
            phiint = int(10*self.noInternalPoints/20)
            tb_start = 1
            tb_fin = int(0.05*self.iterations)
            tint_start = 1
            tint_fin = int(0.05*self.iterations)

            while(it < int(self.iterations)):
        
                if (macroit > 1):
                    phiint = int(10*self.noInternalPoints/20)
                    phib = int(10*self.noBoundaryPoints/20)
                    tb_start = tb_start + 1
                    tb_fin = tb_start + phib -1
                    tint_start = tint_start + 1
                    tint_fin = tint_start + phiint -1

                for tb in range(tb_start, tb_fin):

                    self.trainingOperationBoundary(it)
                    it+=1
                    self.draw1(it)

                for ti in range(tint_start, tint_fin):

                    self.trainingOperationInternal(it)
                    it+=1
                macroit+=1

            #self.plot(it)
            #self.gif()   
            #self.removePng()   
        if(self.training == "batch"):
            

            it = 1
            print("ordering stage batch")
            while(it < int(self.iterations)):

                self.trainingOperationGeneralBatch(it)
                it+=self.batchSize
                self.draw1(it)
    
            print("refinement stage")

            tint_start = 1
            macroit =1

            phib = int(10*self.noBoundaryPoints/20)
            phiint = int(10*self.noInternalPoints/20)
            tb_start = 1
            tb_fin = int(0.05*self.iterations)
            tint_start = 1
            tint_fin = int(0.05*self.iterations)

            while(it < int(self.iterations)):
        
                if (macroit > 1):
                    phiint = int(10*self.noInternalPoints/20)
                    phib = int(10*self.noBoundaryPoints/20)
                    tb_start = tb_start + 1
                    tb_fin = tb_start + phib -1
                    tint_start = tint_start + 1
                    tint_fin = tint_start + phiint -1

                for tb in range(tb_start, tb_fin):

                    self.trainingOperationBoundary(it)
                    it+=1

                for ti in range(tint_start, tint_fin):

                    self.trainingOperationInternal(it)
                    it+=1

                macroit+=1


    def draw1(self, it):

         if(it%1 == 0):
             plt.clf()
             plt.triplot(self.weightsTf[:,0], self.weightsTf[:,1], self.connection) 
             plt.gca().set_aspect('equal', adjustable='box')
             plt.draw()
             plt.pause(0.0001)

    def smoothing(self):

        alpha_prob = self.noInternalPoints/(self.noBoundaryPoints*4 + self.noInternalPoints)

        print("smoothing stage")

        for it in range(1, 10*int(self.iterations)):

            alpha = np.random.uniform(0, 1, 1)

            self.smoothingInternal(it)
            if(alpha > alpha_prob):
                self.smoothingBoundary(it)
            else:
                self.smoothingInternal(it)
            if(it%200 == 0):
                plt.clf()
                plt.triplot(self.weightsTf[:,0], self.weightsTf[:,1], self.connection) 
                plt.gca().set_aspect('equal', adjustable='box')
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1], c = self.lateralConnection) 
                #plt.scatter(self.weightsInner[:,0], self.weightsInner[:,1]) 
                #plt.scatter(self.randomInput[0], self.randomInput[1], color='yellow') 
                #plt.plot(self.geometry[0][:,0], self.geometry[0][:,1], color='red')
                #plt.plot(self.geometry[1][:,0], self.geometry[1][:,1], color='red')
                plt.draw()
                plt.pause(0.0001)

        self.stopTimer("smoothing")

    def moveBoundaryPoints(self):
        """ move boundary weights/points on the geometry boundary """

        self.startTimer()

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

        self.stopTimer("moveBoundaryPoints")



