#This is a Ghaphical user interface for manual annotation of points to generate accurate masks and key-points

from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import json
import random
from tkinter import filedialog
import matplotlib.pyplot as plt
from gui_maskrcnn import Architecture_maskrcnn
from gui_contour import Contour
from gui_keypointrcnn import Architecture_keypointrcnn
from gui_keypoint_canvas import KeyPoints

class Points(Frame):

    def mouseEnter(self,e):
        self.my_canvas.itemconfig(CURRENT,fill=self.colors[int(self.my_canvas.gettags(CURRENT)[1])],activewidth=1,outline='white')
        self.my_canvas.itemconfig(self.tag_text,text="Id:-"+str(self.my_canvas.find_withtag(CURRENT)))
    
    def mouseLeave(self,e):
        self.my_canvas.itemconfig(CURRENT,fill=self.colors[int(self.my_canvas.gettags(CURRENT)[1])])

    def mouseEnter2(self,e):
        self.key_canvas.itemconfig(CURRENT,fill=self.colors[int(self.my_canvas.gettags(CURRENT)[1])],activewidth=1,outline='white')
        self.key_canvas.itemconfig(self.tag_text,text="Id:-"+str(self.key_canvas.find_withtag(CURRENT)))
    
    def mouseLeave2(self,e):
        self.key_canvas.itemconfig(CURRENT,fill=self.colors[int(self.my_canvas.gettags(CURRENT)[1])])
    
    def createPoint(self,xcor,ycor,tag):
        point=self.my_canvas.create_oval(xcor,ycor,xcor+10,ycor+10,fill="green", tag=tag)
        self.my_canvas.tag_bind(point, "<Any-Enter>", self.mouseEnter)
        self.my_canvas.tag_bind(point, "<Any-Leave>", self.mouseLeave)
        return point

    def mouseDown(self,e):
        self.lastx=e.x
        self.lasty=e.y
    
    def reinfor_thresh(self,k):
        prev=self.my_canvas.coords(k-1)
        curr=self.my_canvas.coords(k)
        next=self.my_canvas.coords(k+1)
        pprev=self.my_canvas.coords(k-2)
        nnext=self.my_canvas.coords(k+2)
        px=abs(int(prev[0])-int(curr[0]))
        py=abs(int(prev[1])-int(curr[1]))
        nx=abs(int(next[0])-int(curr[0]))
        ny=abs(int(next[1])-int(curr[1]))
        ppx=abs(int(pprev[0])-int(curr[0]))
        ppy=abs(int(pprev[1])-int(curr[1]))
        nnx=abs(int(nnext[0])-int(curr[0]))
        nny=abs(int(nnext[1])-int(curr[1]))
        return (px+py)/10,(nx+ny)/10,(ppx+ppy)/10,(nnx+nny)/10

    def reinforcement(self,e):
        k=self.my_canvas.find_withtag(CURRENT)[0]
        p,n,pp,nn=self.reinfor_thresh(k=k)
        self.my_canvas.move(CURRENT,e.x-self.lastx,e.y-self.lasty)
        self.my_canvas.move(k-1,((e.x-self.lastx)/p),((e.y-self.lasty)/p))
        self.my_canvas.move(k+1,((e.x-self.lastx)/n),((e.y-self.lasty)/n))
        self.my_canvas.move(k-2,((e.x-self.lastx)/pp),((e.y-self.lasty)/pp))
        self.my_canvas.move(k+2,((e.x-self.lastx)/nn),((e.y-self.lasty)/nn))

    def mouseMove(self,e):
        # print(type(self.my_canvas.find_withtag(CURRENT)[0]))
        # print(self.my_canvas.gettags(CURRENT))
        if self.my_canvas.gettags(CURRENT)[0]=="selected" or self.my_canvas.gettags(CURRENT)[0]=="newPoint":
            # self.reinforcement(e)
            self.my_canvas.move(CURRENT,e.x-self.lastx,e.y-self.lasty)
            if self.my_canvas.gettags(CURRENT)[0]=="selected":
                self.updatePoints(self.my_canvas.gettags(CURRENT)[1])        
        self.lastx=e.x
        self.lasty=e.y

    def mouseMove2(self,e):
        # print(self.my_canvas.gettags(CURRENT))
        self.key_canvas.move(CURRENT,e.x-self.lastx,e.y-self.lasty)
        self.lastx=e.x
        self.lasty=e.y

    def addNewPoint(self,e=None):
        k=self.my_canvas.gettags(int(self.get_the_tag.get()))[1]
        new_point=self.createPoint(20,20,("selected",k))
        pointsArray=self.masksArrayCor[int(k)]
        if new_point not in pointsArray:
            t=int(self.get_the_tag.get())-1
            if int(k)==0:
                pointsArray.insert(t,new_point)
            else:
                sum=0
                for i in range(int(k)):
                    sum=sum+len(self.masksArrayCor[i])
                pointsArray.insert(t-sum,new_point)
            self.masksArrayCor[int(k)]=pointsArray
        self.get_the_tag.delete(0,END)
        # self.get_the_tag.config(state=DISABLED)
    
    def generateColors(self):
        colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        self.colors =[]
        for i in range(self.no_of_masks):
            print(colors[i])
            r, g, b = colors[i]
            self.colors.append(f'#{r:02x}{g:02x}{b:02x}')

    def getCoordinates(self):
        self.all_masks_cor={}
        self.xcor=[]
        self.ycor=[]
        self.all_points_filepath="/Users/aniruddhashadagali/All Codes/PythonCode/GUI/many_people.json"
        with open(self.all_points_filepath,'r') as openfile:
            all_points=json.load(openfile)
        self.no_of_masks=len(all_points)
        self.generateColors()
        for i in range(len(all_points)):
            xcor=all_points["mask "+str(i)]["xcor"][0::20]
            ycor=all_points["mask "+str(i)]["ycor"][0::20]
            x=len(xcor)
            self.theAlgorithm(xcor,ycor,x,i)
    
    def theAlgorithm(self,xcor,ycor,x,p):
        self.threshold=1
        self.newxcor=[]
        self.newycor=[]
        self.newxcor.append(xcor[0])
        self.newycor.append(ycor[0])
        i=0
        while i<x:
            if abs(self.newycor[-1]-ycor[i])>self.threshold and abs(self.newxcor[-1]-xcor[i])>self.threshold:
                self.newxcor.append(xcor[i])
                self.newycor.append(ycor[i])
            i+=1
        self.num_points=len(self.newxcor)
        ddict={"xcor":self.newxcor,"ycor":self.newycor,"length":len(self.newxcor)}
        self.all_masks_cor["mask "+str(p)]=ddict

    def generateAllMasks(self):
        self.masksArrayCor=[]
        for i in range(self.no_of_masks):
            pointsArray=[]
            ddict=self.all_masks_cor["mask "+str(i)]
            x=ddict["xcor"]
            y=ddict["ycor"]
            for j in range(ddict["length"]):
                pointsArray.append(self.createPoint(x[j],y[j],("selected",i)))
            self.masksArrayCor.append(pointsArray)

    def addSegment(self):
        self.segmentMask=self.my_canvas.create_polygon(self.allcor,outline="green", fill="#eeeeff", width=2,state='normal')

    def removeMask(self,e=None):
        self.my_canvas.delete('polygonImage')
        # self.my_canvas.delete(self.newMask)

    def newCoordinates(self,tag):
        pointsArray=self.masksArrayCor[int(tag)]
        new_xcor=[]
        new_ycor=[]
        for i in range(len(pointsArray)):
            point=pointsArray[i]
            new_xcor.append(int(self.my_canvas.coords(point)[0]))
            new_ycor.append(int(self.my_canvas.coords(point)[1]))
        self.num_points=len(new_xcor)
        return new_xcor,new_ycor

    def updatePoints(self,tag):
        x,y=self.newCoordinates(tag)
        newallcor=[]
        for i in range(self.num_points):
            newallcor.append(x[i]+5)
            newallcor.append(y[i]+5)
        newMask=self.createPolygon(newallcor,fill=self.colors[int(tag)], alpha=0.4)
        newallcor.clear()

    def createPolygon(self,*args,**kwargs):
        self.images=[]
        if "alpha" in kwargs:
            if "fill" in kwargs:
                self.fill=self.winfo_rgb(kwargs.pop("fill"))+(int(kwargs.pop("alpha")*255),)
                self.outline=kwargs.pop("outline") if "outline" in kwargs else None
                self.image_of_mask=Image.new("RGBA",(max(args[0][::2]),max(args[0][1::2])))
                ImageDraw.Draw(self.image_of_mask).polygon(args[0], fill=self.fill, outline=self.outline,width=2)
                self.images.clear()
                self.images.append(ImageTk.PhotoImage(self.image_of_mask)) 
                return self.my_canvas.create_image(0, 0, image=self.images[-1], anchor="nw", tag='polygonImage') 
            raise ValueError("fill color must be specified!")
        return self.my_canvas.create_polygon(*args[0], **kwargs,tag='polygon')

    def saveToFile(self):
        print("this will save the final mask")
        # a=[int(x) for x in self.x]
        # b=[int(x) for x in self.y]
        # dictionary = {"xcor":self.new_xcor,"ycor":self.new_ycor}
        # json_object = json.dumps(dictionary, indent=4)
        # with open("final.json", "w") as outfile:
        #     outfile.write(json_object)
        # print("saved to file")

    def createButtons(self):
        self.get_the_tag=Entry(self,width=10, font=('Arial 24'))
        self.get_the_tag.pack(side=LEFT)
        self.btn1=Button(self,text="Add Point",padx=35, pady=10, bg="blue",command=self.addNewPoint)
        self.btn1.pack(side=LEFT)
        self.btn2=Button(self,text="Save Mask",padx=35, pady=10, bg="blue",command=self.saveToFile)
        self.btn2.pack(side=LEFT)
        self.tag_text=self.my_canvas.create_text(50, 50, text="Id:- 0", fill="black", font=('Helvetica 15 bold'))
        self.my_canvas.pack()

    def addImage(self):
        self.resizeImage()
        self.backgroundImage=ImageTk.PhotoImage(self.resize_img)
        self.bgimage=self.my_canvas.create_image(10, 10, image = self.backgroundImage, anchor = NW, tag='image')
    
    def resizeImage(self):
        self.resize_img = Image.open(image_data)
        basewidth = 700
        if self.resize_img.size[0]>=self.resize_img.size[1]:
            wpercent = (basewidth / float(self.resize_img.size[0]))
            hsize = int((float(self.resize_img.size[1]) * float(wpercent)))
            self.resize_img = self.resize_img.resize((basewidth, hsize), Image.ANTIALIAS)
        else:
            wpercent = (basewidth / float(self.resize_img.size[1]))
            hsize = int((float(self.resize_img.size[0]) * float(wpercent)))
            self.resize_img = self.resize_img.resize((hsize, basewidth), Image.ANTIALIAS)

    def inputImgResize(self):
        img = Image.open(image_data)
        basewidth = 200
        if img.size[0]>=img.size[1]:
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        else:
            wpercent = (basewidth / float(img.size[1]))
            hsize = int((float(img.size[0]) * float(wpercent)))
            img = img.resize((hsize,basewidth), Image.ANTIALIAS)
        return img

    def searchImg(self):
        global img, image_data
        image_data = filedialog.askopenfilename(initialdir="/Downloads", title="Choose an image",filetypes=(("all files", "*.*"), ("png files", "*.png"),("HEIC files","*.HEIC"),("jpg files","*.jpg"),("jpeg files","*.jpeg")))
        img=self.inputImgResize()
        img = ImageTk.PhotoImage(img)
        file_name = image_data.split('/')
        self.panel =Label(self, text= file_name[-1])
        self.panel.place(x=300,y=50)
        self.panel_image =Label(self, image=img)
        self.panel_image.place(x=300,y=70)

    def classify(self):
        print("classify the image")
        self.predict_class = Architecture_maskrcnn()
        self.img, self.pred_classes, self.masks = self.predict_class.predImage(image_path=image_data)
        self.classifyData()
    
    def classifyData(self):
        print(self.pred_classes)
        self.arr=[]
        plt.title(label="masked image")
        plt.imshow(self.img)
        plt.show()
        for i in range(len(self.pred_classes)):
            if self.pred_classes[i]=="tie":
                continue
            indices=self.masks[i].astype(np.uint8)
            indices*=255
            self.arr.append(indices)
            plt.title(label=self.pred_classes[i])
            plt.imshow(self.masks[i])
            plt.show()
        k=Contour(mask_array=self.arr)

    def keyPointClassify(self):
        print("key Point classify the image")
        predict_class = Architecture_keypointrcnn()
        self.key_point_img,self.keyOutputs =predict_class.predImage(image_path=image_data)
        self.keyClassifyData()
        # k=KeyPoints(self)
        # k.keyPointClassify(image_path=image_data)
    
    def keyClassifyData(self):
        plt.title(label=" key point image")
        plt.imshow(self.key_point_img)
        plt.show()
        # print(self.key_points)
        self.keyPointAnotate()

    def addKeyImage(self):
        self.resizeImage()
        self.backgroundImage=ImageTk.PhotoImage(self.resize_img)
        self.bgimage=self.key_canvas.create_image(10, 10, image = self.backgroundImage, anchor = NW, tag='image')

    def keyCreatePoint(self,xcor,ycor,tag):
        point=self.key_canvas.create_oval(xcor,ycor,xcor+10,ycor+10,fill="green", tag=tag)
        self.key_canvas.tag_bind(point, "<Any-Enter>", self.mouseEnter2)
        self.key_canvas.tag_bind(point, "<Any-Leave>", self.mouseLeave2)
        return point
    
    def generateKeyPoints(self):
        x=[]
        y=[]
        for i in range(len(self.keyOutputs[0]['keypoints'])):
                keypoints = self.keyOutputs[0]['keypoints'][i].cpu().detach().numpy()
                if self.keyOutputs[0]['scores'][i] > 0.9: 
                    keypoints = keypoints[:, :].reshape(-1, 3)
                    for p in range(keypoints.shape[0]):
                        x.append(int(keypoints[p,0]))
                        y.append(int(keypoints[p, 1]))
        self.keyPoints=[]
        for j in range(len(x)):
            self.keyPoints.append(self.keyCreatePoint(x[j],y[j],("selected",i)))

    def keyPointAnotate(self):
        for img_display in self.winfo_children():
            img_display.destroy()
        self.title =Label(self, text="Key-Point-Annotation", padx=25, pady=6, font=("", 12)).pack()
        self.key_canvas=Canvas(self, width="600",height="600",bg="white")
        self.key_canvas.pack()
        self.addKeyImage()
        self.generateKeyPoints()
        get_the_tag=Entry(self,width=10, font=('Arial 24'))
        get_the_tag.pack(side=LEFT)
        btn1=Button(self,text="Add Point",padx=35, pady=10, bg="blue",command=self.addNewPoint)
        btn1.pack(side=LEFT)
        btn2=Button(self,text="Save KeyPoints",padx=35, pady=10, bg="blue",command=self.saveToFile)
        btn2.pack(side=LEFT)
        tag_text=self.key_canvas.create_text(50, 50, text="Id:- 0", fill="black", font=('Helvetica 15 bold'))
        self.key_canvas.pack()
        Widget.bind(self.key_canvas, "<1>", self.mouseDown)
        Widget.bind(self.key_canvas, "<B1-Motion>", self.mouseMove2)

    def manualAnotate(self):
        print("maual anotate")

    def createCanvas(self):
        self.getCoordinates()
        for img_display in self.winfo_children():
            img_display.destroy()
        self.title =Label(self, text="Semi-Annotation", padx=25, pady=6, font=("", 12)).pack()
        self.my_canvas=Canvas(self, width="700",height="700",bg="white")
        self.my_canvas.pack()
        self.addImage()
        self.generateAllMasks()
        self.createButtons()
        Widget.bind(self.my_canvas, "<1>", self.mouseDown)
        Widget.bind(self.my_canvas, "<B1-Motion>", self.mouseMove)
        self.bind_all("<Button-2>",self.removeMask)
        self.bind_all("<Key-v>",self.removeMask)
        self.bind_all("<Return>",self.addNewPoint)

    def loadImage(self):
        self.title =Label(self, text="Annotation tool", padx=25, pady=6, font=("", 12)).pack()
        self.canvas =Canvas(self, height=700, width=700, bg='white').pack() 
        self.choose_image=Button(self, text='Choose Image',padx=35, pady=10, bg="blue", command=self.searchImg)
        self.choose_image.pack(side=LEFT)
        self.classify_image=Button(self, text='Classify Image',padx=35, pady=10, bg="blue", command=self.classify)
        self.classify_image.pack(side=LEFT)
        self.semi_anotate=Button(self, text='Semi Anotate',padx=35, pady=10, bg="blue", command=self.createCanvas)
        self.semi_anotate.pack(side=LEFT)
        self.manual_anotate=Button(self, text='Manual Anotate',padx=35, pady=10, bg="blue", command=self.manualAnotate)
        self.manual_anotate.pack(side=LEFT)
        self.manual_anotate=Button(self, text='Key-Point Anotate',padx=35, pady=10, bg="blue", command=self.keyPointClassify)
        self.manual_anotate.pack(side=LEFT)

    def homePageCommand(self):
        for img_display in self.winfo_children():
            img_display.destroy()
        self.loadImage()

    def __init__(self,master):
        Frame.__init__(self, master)
        self.master.title("Full Anotation tool")
        Pack.config(self)
        self.home_button=Button(self.master, text='Home Page',padx=35, pady=10, bg="blue", command=self.homePageCommand)
        self.home_button.place(x=10, y=0)
        self.loadImage()
t=Tk()
test=Points(t)
test.mainloop()