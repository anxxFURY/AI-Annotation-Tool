from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import json
from tkinter import filedialog
import matplotlib.pyplot as plt
from gui_maskrcnn import Architecture_maskrcnn
import cv2

class ImageInput(Frame):

    def searchImg(self):
        global img, image_data
        image_data = filedialog.askopenfilename(initialdir="/Downloads", title="Choose an image",filetypes=(("all files", "*.*"), ("png files", "*.png"),("jpg files","*.jpg")))
        basewidth = 150 
        img = Image.open(image_data)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        file_name = image_data.split('/')
        self.panel =Label(self.canvas, text= file_name[-1])
        self.panel.place(x=300,y=50)
        self.panel_image =Button(self.canvas, image=img,padx=35, pady=10, bg="blue", command=self.matplot)
        self.panel_image.place(x=300,y=70)

    def matplot(self):
        g=Image.open(image_data)
        plt.imshow(g)
        plt.show()
    
    def createCanvas(self):
        for img_display in self.winfo_children():
            img_display.destroy()
        self.my_canvas=Canvas(self, width="800",height="700",bg="lightgreen")
        self.my_canvas.pack(pady=20)

    def classify(self):
        for img_display in self.winfo_children():
            img_display.destroy()
        self.predict_class = Architecture_maskrcnn()
        self.img, self.pred_classes, self.masks = self.predict_class.predImage(image_path=image_data)
        self.classifyData()
        # self.plotOnCanvas()
        
    def plotOnCanvas(self):
        m=20
        n=150
        basewidth=150
        wpercent = (basewidth / float(self.masks[0].shape[1]))
        hsize = int((float(self.masks[0].shape[0]) * float(wpercent)))
        # while(i+j<len(self.pred_classes)):
        for i in range(4):
            for j in range(4):
                self.panel =Label(self.canvas, text= "sfsd")
                self.panel.place(x=m,y=n-20)
                # r=cv2.resize(self.masks[0], (int(basewidth),int( hsize)),interpolation = cv2.INTER_LINEAR)
                new_image_mask = np.expand_dims(self.masks[0],-1)*np.ones((1,1,3))
                # k = ImageTk.PhotoImage(new_image_mask)
                image_button=Button(self.canvas, image=new_image_mask,padx=35, pady=10, bg="blue", command=self.matplot)
                image_button.place(x=m,y=n)
                m=m+170
            m=20
            n=n+150

    def classifyData(self):
        print(self.pred_classes)
        self.arr=[]
        plt.title(label="masked image")
        plt.imshow(self.img)
        plt.show()
        for i in range(len(self.pred_classes)):
            indices=self.masks[i].astype(np.uint8)
            indices*=255
            self.arr.append(indices)
            plt.title(label=self.pred_classes[i])
            plt.imshow(self.masks[i])
            plt.show()
        self.resizeImage(self.masks[0])

    def resizeImage(self,image):
        plt.imshow(image)
        plt.show()
        # self.resize_img=image
        # basewidth = 750 
        # wpercent = (basewidth / float(self.resize_img.shape[0]))
        # hsize = int((float(self.resize_img.size[1]) * float(wpercent)))
        # self.resize_img = self.resize_img.resize((basewidth, hsize), Image.ANTIALIAS)
        # plt.imshow(self.resize_img)
        # plt.show()

    def loadImage(self):
        self.title =Label(self.master, text="Annotation tool", padx=25, pady=6, font=("", 12)).pack()
        self.canvas =Canvas(self.master, height=700, width=700, bg='red').pack(expand='yes',fill='both') 
        self.choose_image=Button(self.master, text='Choose Image',padx=35, pady=10, bg="blue", command=self.searchImg)
        self.choose_image.pack(side=LEFT)
        self.classify_image=Button(self.master, text='Classify Image',padx=35, pady=10, bg="blue", command=self.classify)
        self.classify_image.pack(side=LEFT)
        # self.classify_image=Button(self.master, text='Csdfs Image',padx=35, pady=10, bg="blue", command=self.classify)
        # self.classify_image.pack(side=LEFT)

    def __init__(self,master):
        Frame.__init__(self, master)
        Pack.config(self)
        self.loadImage()
        
root=Tk()
root.title("mask tool")
first_page=ImageInput(root)
first_page.mainloop()