from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageSelector(Frame):

    def mainnn(self):
        self.canvas =Canvas(self, width=500, height=500)
        self.canvas.pack(side=LEFT)
        self.image_select_btn = Button(self,text="Select image",command=self.select_image)
        self.image_select_btn.pack(side=LEFT)
        
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)
            
    def on_move_press(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)
        
    def on_button_release(self, event):
        cropped_image = self.image.crop((self.start_x, self.start_y, self.end_x, self.end_y))
        self.display_cropped_image(cropped_image)
        
    def display_cropped_image(self, cropped_image):
        cropped_image_tk = ImageTk.PhotoImage(cropped_image)
        cropped_window = Toplevel()
        cropped_canvas = Canvas(cropped_window, width=cropped_image.width, height=cropped_image.height)
        cropped_canvas.pack()
        cropped_canvas.create_image(0, 0, anchor=NW, image=cropped_image_tk)
        cropped_window.mainloop()


    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        self.image = Image.open(self.image_path)
        self.image.resize((500,500))
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)


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
        self.mainnn()
        # self.loadImage()
        # print("hellow")
t=Tk()
t.title("custom annotation tool")
test=ImageSelector(t)
test.mainloop()