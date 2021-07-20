"""
Streamlit
Entorno virtual Streamlit_app / Python 3.8

pip install streamlit
pip install opencv-python
pip install -U scikit-learn
pip install matplotlib

"""
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import urllib
import cv2
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import random
from PIL import Image

def upload_image(selection):
     if selection == "Cargar imagen propia":
         filename = st.sidebar.file_uploader("Seleccionar archivo", type=["png", "jpg", "jpeg"])
     else: 
         filename = "zidane.jpg"
     return filename

def main():
     st.sidebar.title("Detección de Objetos")
     image_selection = st.sidebar.selectbox("Seleccionar imagen",["Cargar imagen propia", "Usar imagen de ejemplo"])
     image_filename = upload_image(image_selection)
     run_detection(image_filename)
# Cached function that returns a mutable object with a random number in the range 0-100
@st.cache(allow_output_mutation=True)
def seed():
     return {'seed': random.randint(0, 100)} # Mutable (dict)


def run_detection(file_uploaded):
     st.header("Detección YOLO v3 app")

     if file_uploaded is not None:
         image = Image.open(file_uploaded)
         st.set_option('deprecation.showPyplotGlobalUse', False)
         column1, column2 = st.beta_columns(2)
         # Localise detected objects in the image
         column1.subheader("Input image")
         st.text("")
         # Display the input image using matplotlib
         plt.figure(figsize=(16, 16))
         plt.imshow(image)
         column1.pyplot(use_column_width=True)
         neural_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
         labels = [] # Initialize an array to store output labels
         with open("coco.names", "r") as file:
             labels = [line.strip() for line in file.readlines()]
             names_of_layer = neural_net.getLayerNames()
             output_layers = [names_of_layer[i[0]-1] for i in neural_net.getUnconnectedOutLayers()]
             colors = np.random.uniform(0, 255, size=(len(labels), 3))
             newImage = np.array(image.convert('RGB'))
             img = cv2.cvtColor(newImage, 1)
             height, width, channels = img.shape
             # Convert the images into blobs
             blob = cv2.dnn.blobFromImage(img, 0.00391, (416, 416), (0, 0, 0), True, crop=False)
             neural_net.setInput(blob) # Feed the model with blobs as the input
             outputs = neural_net.forward(output_layers)
             classID = [] 
             confidences = [] 
             boxes = []
             # Add sliders for confidence threshold and NMS threshold in the sidebar
             score_threshold = st.sidebar.slider("Confidence_threshold",0.00, 1.00, 0.5, 0.01)
             nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.5,0.01)
             # Localise detected objects in the image
             for op in outputs:
                 for detection in op:
                     scores = detection[5:]
                     class_id = np.argmax(scores)
                     confidence = scores[class_id] 
                     if confidence > 0.5:
                         center_x = int(detection[0] * width)
                         center_y = int(detection[1] * height) # centre of object
                         w = int(detection[2] * width)
                         h = int(detection[3] * height)
                         # Calculate coordinates of bounding box
                         x = int(center_x - w / 2)
                         y = int(center_y - h/2)
                         # Organize the detected objects in an array
                         boxes.append([x,y,w,h])
                         confidences.append(float(confidence))
                         classID.append(class_id)
            
             indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
             # assign color to differente objects
             items = []

             for i in range(len(boxes)):
                 if i in indexes:
                     x,y,w,h = boxes[i]
                     label = str.upper((labels[classID[i]]))
                     color = colors[i]
                     cv2.rectangle(img, (x, y), (x+w, y+h),color, 3)
                     items.append(label)
             st.text("")
             column2.subheader("Output image")
             st.text("")
             # Display the input image using matplotlib
             plt.figure(figsize=(15, 15))
             plt.imshow(img)
             column2.pyplot(use_column_width=True)

             if len(indexes) > 1:
                 st.success("Se encontraron {} Objectos -{}".format(len(indexes), [item for item in set(items)]))

             else:
                 st.success("Se encontraron {} Objectos -{}".format(len(indexes), [item for item in set(items)]))
             
                 
                 
             
    
     

if __name__=='__main__':
    main()
    
