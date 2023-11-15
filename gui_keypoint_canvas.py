
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

class KeyPoints(Frame):

    def mouseEnter2(self,e):
        self.key_canvas.itemconfig(CURRENT,fill='red',activewidth=1,outline='white')
    
    def mouseLeave2(self,e):
        self.key_canvas.itemconfig(CURRENT,fill='green')

    def mouseDown(self,e):
        self.lastx=e.x
        self.lasty=e.y

    def mouseMove2(self,e):
        # print(self.key_canvas.gettags(CURRENT))
        if self.key_canvas.gettags(CURRENT)[0]=="selected":
            self.key_canvas.move(CURRENT,e.x-self.lastx,e.y-self.lasty)
        self.lastx=e.x
        self.lasty=e.y

    def saveToFile(self):
        print("this will save the final mask")
        # a=[int(x) for x in self.x]
        # b=[int(x) for x in self.y]
        # dictionary = {"xcor":self.new_xcor,"ycor":self.new_ycor}
        # json_object = json.dumps(dictionary, indent=4)
        # with open("final.json", "w") as outfile:
        #     outfile.write(json_object)
        # print("saved to file")


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
        image_data = filedialog.askopenfilename(initialdir="/Downloads", title="Choose an image",filetypes=(("all files", "*.*"), ("png files", "*.png"),("jpg files","*.jpg"),("jpeg files","*.jpeg")))
        img=self.inputImgResize()
        img = ImageTk.PhotoImage(img)
        file_name = image_data.split('/')
        self.panel =Label(self, text= file_name[-1])
        self.panel.place(x=300,y=50)
        self.panel_image =Label(self, image=img)
        self.panel_image.place(x=300,y=70)

    def keyPointClassify(self,image_path):
        print("key Point classify the image")
        global img, image_data
        image_data=image_path
        predict_class = Architecture_keypointrcnn()
        self.key_point_img,self.keyOutputs =predict_class.predImage(image_path=image_path)
        self.keyClassifyData()
    
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
        self.key_canvas=Canvas(self, width="800",height="750",bg="lightblue")
        self.key_canvas.pack()
        self.addKeyImage()
        self.generateKeyPoints()
        get_the_tag=Entry(self,width=10, font=('Arial 24'))
        get_the_tag.pack(side=LEFT)
        # btn1=Button(self,text="Add Point",padx=35, pady=10, bg="blue",command=self.addNewPoint)
        # btn1.pack(side=LEFT)
        btn2=Button(self,text="Save KeyPoints",padx=35, pady=10, bg="blue",command=self.saveToFile)
        btn2.pack(side=LEFT)
        tag_text=self.key_canvas.create_text(50, 50, text="Id:- 0", fill="black", font=('Helvetica 15 bold'))
        self.key_canvas.pack()
        Widget.bind(self.key_canvas, "<1>", self.mouseDown)
        Widget.bind(self.key_canvas, "<B1-Motion>", self.mouseMove2)
        Widget.bind(self.key_canvas, "<1>", self.mouseDown)

    def loadImage(self):
        self.title =Label(self, text="Annotation tool", padx=25, pady=6, font=("", 12)).pack()
        self.canvas =Canvas(self, height=700, width=700, bg='red').pack() 
        self.choose_image=Button(self, text='Choose Image',padx=35, pady=10, bg="blue", command=self.searchImg)
        self.choose_image.pack(side=LEFT)
        self.manual_anotate=Button(self, text='Key-Point Anotate',padx=35, pady=10, bg="blue", command=self.keyPointClassify)
        self.manual_anotate.pack(side=LEFT)

    def __init__(self,master=None):
        Frame.__init__(self, master=None)
        self.master.title("Full Anotate tool")
        Pack.config(self)
        self.loadImage()
# t=Tk()
# test=KeyPoints(t)
# test.mainloop()