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

import time
import matplotlib.pyplot as plt
#import sys
#import imageio
#import glob, os, re

""" defenition of a few helper classes and functions """

class Timer:
    """ measurement of the duration of the called functions """

    def __init__(self):
        """ initialization of the timer """

        self.startTime = dict()
        self.timerSummary = dict()

    def startTimer(self, functionName):
        """ save the start time 

            :param functionName: name of the measured function as string
        """

        self.startTime[functionName] = time.time()

    def stopTimer(self, functionName):
        """ stop the timer and measure 

            :param functionName: name of the measured function as string
        """

        if functionName in self.timerSummary:

            self.timerSummary[functionName] += time.time() - self.startTime[functionName]
        else:
            self.timerSummary[functionName] = time.time() - self.startTime[functionName]

    def printTimerSummary(self):
        """ print the consumed time """

        print("_________________________________________________________")
        print("                        ")
        print("Timer")
        print("_________________________________________________________")
        for x in self.timerSummary:
            print(x, ": ", self.timerSummary[x])
        print("_________________________________________________________")

class Plotter:
    """ plotting of the mesh """

    def __init__(self, path, outputName):
        """ initialization of the timer 

            :param path: name of the folder where the output is saved
            :param outputName: naming of the output
        """

        self.path = path
        self.outputName = outputName

    def plot(self,it, x, y, con, png = False):
        """ save figure as png or plot figure """

        if(png):

            fig, ax = plt.subplots() 
            plt.triplot(self.weightsTf[:,0], self.weightsTf[:,1], self.connection)
            plt.gca().set_aspect('equal', adjustable='box')

            name = '/naca_' + str(it) + '.png'

            fig.savefig(name)
            plt.close(fig)
        else:

            plt.clf()
            plt.triplot(self.weightsTf[:,0], self.weightsTf[:,1], self.connection) 
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()
            plt.pause(0.0001)


    def sorted_alphanumeric(self, data):
        """ numerical sort of the data files in folder """

        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def gif(self):
        """ produce a gif from the png files """

        dir_name = "pics/"
        filenames = self.sorted_alphanumeric(os.listdir(dir_name))
        print(filenames)

        with imageio.get_writer('pics/naca.gif', mode='I', fps=20) as writer:
            for filename in filenames:

                if filename.endswith(".png"):
                    image = imageio.imread(dir_name + filename)
                    writer.append_data(image)

    def removePng(self):
        """ delete all pngs in folder """

        dir_name = "pics/"
        filenames = os.listdir(dir_name)

        for filename in filenames:
            if filename.endswith(".png"):
                os.remove(os.path.join(dir_name, filename))




def calculateGridQuality():
    """ calculate the quality of the mesh """

    cellAspectRatio = None
    skewness = None
    orthogonality = None
    smoothness = None




