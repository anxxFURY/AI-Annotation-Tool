import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import json

class Contour:
    
    def processImage(self,mask):
        self.img =mask
        # self.img = self.img_path
        # self.img = cv2.imread('/Users/surajreddy/Downloads/cevi/maskrcnn/img5/Screenshot 2022-12-22 at 11.05.01 AM.png')
        # if self.img.shape[0]>=800 or self.img.shape[1]>=700:
        #     self.img=cv2.resize(self.img, (int(self.img.shape[1]/2),int( self.img.shape[0]/2)),interpolation = cv2.INTER_LINEAR)
        # print(self.img.shape)
        # self.imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # self.ret, self.thresh = cv2.threshold(self.imgray, 127, 255, 0)
        self.contours, self.hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # self.contours, self.hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.no_of_contours=len(self.contours)
        print("Number of contours = " + str(len(self.contours)))
        cv2.drawContours(self.img, self.contours,-1, (0, 255, 0), 3)
        # cv2.drawContours(self.imgray, self.contours, -1, (0, 255, 0), 3)

    def plotContours(self):
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(2, 2, 1)
        plt.imshow(self.img)
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(self.imgray)
        plt.show()
    
    def plotInMatplotLib(self,l):
        self.x=[]
        self.y=[]
        for i in self.contours[0]:
            self.x.append(i[0][0])
            self.y.append(i[0][1])
        # xpoints = np.array(self.x)
        # ypoints = np.array(self.y)
        # plt.plot(xpoints, -ypoints)
        # plt.show()
        a=[int(x) for x in self.x]
        b=[int(x) for x in self.y]
        dictionary = {"xcor":a,"ycor":b}
        self.dictionary[l]=dictionary
        # a.clear()
        # b.clear()
        self.x.clear()
        self.y.clear()
    
    def saveToFile(self):
        json_object = json.dumps(self.dictionary, indent=4)
        with open("many_people.json", "w") as outfile:
            outfile.write(json_object)
        print("saved to file")

    def __init__(self,mask_array):
        self.mask_array=mask_array
        self.masks_length=len(self.mask_array)
        self.dictionary={}
        for i in range(self.masks_length):
            print("get the contour points of mask "+ str(i))
            l="mask "+str(i)
            self.processImage(self.mask_array[i])
            # self.plotContours()
            self.plotInMatplotLib(l)
        self.saveToFile()
        
# k=Contour('/Users/surajreddy/Downloads/cevi/maskrcnn/img5/Screenshot 2022-12-22 at 11.05.01 AM.png')





