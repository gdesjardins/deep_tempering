import matplotlib.pyplot as plot
import matplotlib
import cPickle
import numpy



def loadData(filename):
    data = numpy.load(filename)
    #N = 100000
    #data = data[0:N,]
    #data = data/data[0]
    #print data
    #data = cPickle.load(open(filename, "r"))
    return data
   
   
def plotData(data, plotName, lineType = '-'):
    
    # Each item in data is a set of points to plot
    plot.plot(range(len(data)), data, lineType, label=plotName)
    
    
def outputPlot(outputFilename):
    plot.xlabel('lag')
    plot.ylabel('autocorrelation')
    plot.grid(True)
    #plot.legend(loc='upper right')
    plot.legend(loc='best')
    #plot.xticks([])
    #plot.yticks([0,0.8,1.0,1.1])
    plot.yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2])
    plot.show()
    plot.savefig(outputFilename)
    
   
plotNames = ["mnist_rbm_auto",
             "mnist_tdbn2_auto",
             "mnist_tdbn3_auto"]
             
data = loadData(plotNames[0] + ".npy")
plotData(data.mean(axis=0), "RBM", lineType = ':')  

data = loadData(plotNames[1] + ".npy")
plotData(data.mean(axis=0), "t-DBN2", lineType = '-.')    

data = loadData(plotNames[2] + ".npy")
plotData(data.mean(axis=0), "t-DBN3", lineType = '-')    

outputPlot('mnist_auto.png')
