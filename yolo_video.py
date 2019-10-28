import numpy 
import argparse
import imutils
import time
import cv2
import os

LABELS = open("yolo-coco/coco.names").read().strip().split("\n")
numpy.random.seed(42)
COLORS = numpy.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

class Detect:
    def __init__(self):
        global net
        net = cv2.dnn.readNetFromDarknet('yolo-coco/yolov3.cfg', 'yolo-coco/yolov3.weights')
    
    def detectObject(self, imName):
        arr1={}
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        print(imName)
        arg=vars(imName)
        print("videos/"+str(arg['filename']))
        vs = cv2.VideoCapture("videos/"+str(arg['filename']))
        writer = None
        (W, H) = (None, None)
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                   else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
        except:
            total = -1
        c=0
        while (c<5):
            c=c+1
            (grabbed, frame) = vs.read()
            print(grabbed)
            if not grabbed:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = numpy.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[0:4] * numpy.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                    arr1[LABELS[classIDs[i]]]=confidences[i]
                    cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter("videos/"+str(arg['filename'])+".avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
                print("[INFO] Initialized successfully")
            writer.write(frame)
        
        print("[INFO] FILE LOADED SUCCESSFULLY!!!")
        writer.release()
        vs.release()
        return(arr1)
