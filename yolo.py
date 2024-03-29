import numpy 
import time
import cv2
import os

LABELS = open("yolo-coco/coco.names").read().strip().split("\n")
numpy.random.seed(42)
COLORS = numpy.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

class Detector:
    def __init__(self):
        global net
        net = cv2.dnn.readNetFromDarknet('yolo-coco/yolov3.cfg', 'yolo-coco/yolov3.weights')
    
    def detectObject(self, imName):
        arr={}
        image = cv2.cvtColor(numpy.array(imName), cv2.COLOR_BGR2RGB)
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        start=time.time()
        layerOutputs=net.forward(ln)
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                arr[LABELS[classIDs[i]]]=confidences[i]
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        cv2.imshow("detection",image)
        cv2.waitKey(delay=5000)
        cv2.destroyAllWindows()
        return (image,arr)
