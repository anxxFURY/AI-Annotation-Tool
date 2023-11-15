import matplotlib.pyplot as plt
import json

class ReducePoints:

    def getCoordinates(self):
        self.xcor=[]
        self.ycor=[]
        self.all_points_filepath="/Users/surajreddy/Downloads/cevi/giraf.json"
        with open(self.all_points_filepath,'r') as openfile:
            all_points=json.load(openfile)
        self.xcor=all_points["xcor"]
        self.ycor=all_points["ycor"]
        self.num_points=len(self.xcor)
        self.theAlgorithm()
        self.allCoordinates()
        self.plotOnMask()

    def theAlgorithm(self):
        self.threshold=10
        self.newxcor=[]
        self.newycor=[]
        self.newxcor.append(self.xcor[0])
        self.newycor.append(self.ycor[0])
        i=0
        while i<self.num_points:
            if abs(self.newycor[-1]-self.ycor[i])>self.threshold and abs(self.newxcor[-1]-self.xcor[i])>self.threshold:
                self.newxcor.append(self.xcor[i])
                self.newycor.append(self.ycor[i])
            i+=1
        self.num_points=len(self.newxcor)

    def plotOnMask(self):
        im = plt.imread('/Users/surajreddy/Downloads/cevi/maskrcnn/img5/pexels-magda-ehlers-1319515 copy.jpg')
        fig = plt.figure(figsize=(10, 10))
        implot = plt.imshow(im)
        for p,q in zip(self.newxcor,self.newycor):
            x_cord = p
            y_cord = q
            plt.scatter([x_cord], [y_cord])
        plt.show()  

    def allCoordinates(self):
        self.allcor=[]
        for i in range(self.num_points):
            self.allcor.append(self.xcor[i])
            self.allcor.append(self.ycor[i])
    
    def __init__(self):
        self.getCoordinates()

# test=ReducePoints()
