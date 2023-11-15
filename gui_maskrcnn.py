import torch 
import torchvision
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
import requests
from io import BytesIO
import random
import matplotlib.pyplot as plt
from urllib.request import urlopen
import cv2
import json
from gui_contour import Contour
import torchvision.models.detection as detection  

class Architecture_maskrcnn:

  def loadModel(self):
    weights= detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    self.model = detection.maskrcnn_resnet50_fpn(weights=weights)
    # self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # print(model)
    self.model.eval()

  def CocoDatasetClasses(self):
    self.COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    # print(len(self.COCO_INSTANCE_CATEGORY_NAMES))

  def inputImgResize(self,img_path):
      img = Image.open(img_path)
      if img.size[0]>=img.size[1]:
          wpercent = (self.basewidth / float(img.size[0]))
          hsize = int((float(img.size[1]) * float(wpercent)))
          img = img.resize((self.basewidth, hsize), Image.ANTIALIAS)
      else:
          wpercent = (self.basewidth / float(img.size[1]))
          hsize = int((float(img.size[0]) * float(wpercent)))
          img = img.resize((hsize,self.basewidth), Image.ANTIALIAS)
      return img  

  def getPrediction(self,img_path, threshold=0.5, url=False):
    
    if url: # We have to request the image
      response = requests.get(img_path)
      img = Image.open(BytesIO(response.content))
    else:
        img =self.inputImgResize(img_path=img_path)
    transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
    img = transform(img)
    # img = img.cuda() # Only if GPU, otherwise comment this line
    pred = self.model([img]) # Send the image to the model. This runs on CPU, so its going to take time
    #Let's change it to GPU
    # pred = pred.cpu() # We will just send predictions back to CPU
    # Now we need to extract the bounding boxes and masks
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

  def url_to_image(self,url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url) # We want to convert URL to cv2 image here, so we can draw the mask and bounding boxes
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image

  def randomColorMasks(self,image):
    # I will copy a list of colors here
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image==1], g[image==1], b[image==1] = colors[random.randint(1,10)]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask

  def inputImgResizecv2(self,img_path):
    img = cv2.imread(img_path)
    if img.shape[0]<=img.shape[1]:
      wpercent = (self.basewidth / float(img.shape[1]))
      hsize = int((float(img.shape[0]) * float(wpercent))) 
      img=cv2.resize(img, (int(self.basewidth),int( hsize)),interpolation = cv2.INTER_LINEAR)
    else:
      wpercent = (self.basewidth / float(img.shape[0]))
      hsize = int((float(img.shape[1]) * float(wpercent))) 
      img=cv2.resize(img, (int( hsize),int(self.basewidth)),interpolation = cv2.INTER_LINEAR)
    return img

  def instanceSegmentation(self,img_path, threshold=0.5, rect_th=1,text_size=1, text_th=1, url=False):
    masks, boxes, pred_cls = self.getPrediction(img_path, threshold=threshold, url=url)
    if url:
      img = self.url_to_image(img_path) # If we have a url image
    else: # Local image
      img = self.inputImgResizecv2(img_path=img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR
    # plt.imshow(masks[0])
    for i in range(len(masks)):
      rgb_mask = self.randomColorMasks(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      pt1 = tuple(int(x) for x in boxes[i][0])
      pt2 = tuple(int(x) for x in boxes[i][1])
      cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img, pred_cls, masks

  # def saveImages(self):
  #   # cv2.imwrite('./bottle_mask.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  #   # plt.imsave('imgd.png', img)
  #   plt.imshow(masks[0])
  #   # plt.imshow(total_mask)
  #   plt.imsave('horse_.png', masks[0])

  def predImage(self,image_path):
    self.basewidth = 700 
    self.image_path=image_path
    self.loadModel()
    self.CocoDatasetClasses()
    self.img, self.pred_classes, self.masks = self.instanceSegmentation(img_path=self.image_path, rect_th=5, text_th=4)
    return self.img, self.pred_classes, self.masks

  def __init__(self):
    os.environ['TORCH_HOME'] = '/Users/aniruddhashadagali/All Codes/PythonCode/GUI/weights'
    # self.predImage(image_path='/Users/surajreddy/Downloads/pexels-jopwell-2422290.jpg')

# test=Architecture_maskrcnn()
