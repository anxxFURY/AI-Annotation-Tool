import matplotlib.pyplot as plt
import cv2
import matplotlib
import numpy
import torchvision
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
import os

class Architecture_keypointrcnn:

    def get_model(self,min_size=800):
        # initialize the model
        weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights,pretrained=True,num_keypoints=17, min_size=min_size)
        return model

    def show_image(self,path):
        image = plt.imread(path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    def draw_keypoints_and_boxes(self,outputs, image):
        # the `outputs` is list which in-turn contains the dictionary 
        for i in range(len(outputs[0]['keypoints'])):
            # get the detected keypoints
            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
            # get the detected bounding boxes
            boxes = outputs[0]['boxes'][i].cpu().detach().numpy()

            # proceed to draw the lines and bounding boxes 
            if outputs[0]['scores'][i] > 0.9: # proceed if confidence is above 0.9
                keypoints = keypoints[:, :].reshape(-1, 3)
                self.keypoints=keypoints
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                                4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # draw the lines joining the keypoints
                for ie, e in enumerate(self.edges):
                    # get different colors for the edges
                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(self.edges)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                            (int(keypoints[e, 0][1]),int( keypoints[e, 1][1])),
                            tuple(rgb), 5, lineType=cv2.LINE_AA)

                # draw the bounding boxes around the objects
                cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                                color=(0, 255, 0), 
                                thickness=2)
            else:
                continue
        return image
    
    def resizeImage(self,image_path):
        self.resize_img = Image.open(image_path).convert('RGB')
        basewidth = 700
        if self.resize_img.size[0]>=self.resize_img.size[1]:
            wpercent = (basewidth / float(self.resize_img.size[0]))
            hsize = int((float(self.resize_img.size[1]) * float(wpercent)))
            self.resize_img = self.resize_img.resize((basewidth, hsize), Image.ANTIALIAS)
        else:
            wpercent = (basewidth / float(self.resize_img.size[1]))
            hsize = int((float(self.resize_img.size[0]) * float(wpercent)))
            self.resize_img = self.resize_img.resize((hsize, basewidth), Image.ANTIALIAS)

    
    def predImage(self,image_path):
        self.resizeImage(image_path)
        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]
        transform = transforms.Compose([transforms.ToTensor()])
        # set the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the modle on to the computation device and set to eval mode
        model = self.get_model().to(device).eval()
        # image = Image.open(image_path).convert('RGB')
        image=self.resize_img
        # NumPy copy of the image for OpenCV functions
        orig_numpy = np.array(image, dtype=np.float32)
        # convert the NumPy image to OpenCV BGR format
        orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
        # transform the image
        image = transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the detections, forward pass the image through the model
        with torch.no_grad():
            outputs = model(image)

        # draw the keypoints, lines, and bounding boxes
        output_image = self.draw_keypoints_and_boxes(outputs, orig_numpy)
        # visualize the image
        # cv2.imshow("key point image",output_image)
        # cv2.waitKey(0)
        rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        rgb_image *= 255.
        rgb_image = rgb_image.astype(np.uint8)
        pil_image = Image.fromarray(rgb_image)
        # pil_image.show()
        # cv2.imwrite("/Users/surajreddy/Downloads/cevi/GUI/ddf.jpg", output_image*255.)
        # self.show_image("/Users/surajreddy/Downloads/cevi/GUI/ddf.jpg")
        return rgb_image,outputs 
    
    def __init__(self):
        os.environ['TORCH_HOME'] = '/Users/aniruddhashadagali/All Codes/PythonCode/GUI/weights'
        # self.predImage(image_path="/Users/surajreddy/Downloads/cevi/GUI/f.jpg")

# test=Architecture_keypointrcnn()





