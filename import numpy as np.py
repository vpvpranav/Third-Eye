import numpy as np
import cv2
import pyttsx3

prototxt_path =r"C:\Users\vpvpr\Downloads\MobileNetSSD_deploy.prototxt"
model_path =r"C:\Users\vpvpr\Downloads\MobileNetSSD_deploy.caffemodel"
min_confidence = 0.2

classes =['background','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

np.random.seed(543210)
colors = np.random.uniform(0,255,size=(len(classes),3))


net = cv2.dnn.readNet(prototxt_path,model_path)

cap = cv2.VideoCapture(0)
while True:
    _,image= cap.read()
    height , width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007,(300,300),130)

    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):

        confidence =(detected_objects[0][0][i][2])
    
        if confidence>min_confidence:
            class_index = int(detected_objects[0][0][i][1])


            pred_txt = f"{classes[class_index]}"
            engine = pyttsx3.init()
            engine.say(pred_txt)
            engine.runAndWait()


    

cv2.destroyAllWindows()
cap.release()
