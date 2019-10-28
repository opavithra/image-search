from yolo import Detector
from yolo_video import Detect
from IMG import Detection
import io
import cv2

from base64 import b64encode
from flask import Flask, render_template, request

from PIL import Image
from flask import send_file

app = Flask(__name__)

detector = Detector()
detect = Detect()
live    =   Detection()

# detector.detectNumberPlate('twocar.jpg')


@app.route("/")
def index():
        return render_template('home.html')


@app.route("/", methods=['POST'])
def upload():
        if request.method == 'POST':
                if request.form["submit_button"] == "image":
                        file = Image.open(request.files['file'].stream)
                        img,arr= detector.detectObject(file)
                        return render_template('search.html',value=arr,image=img)
                
                
                elif request.form["submit_button"] == "video":
                        f=request.files["file"]
                        arr1=detect.detectObject(f)
                        return render_template('search.html',value=arr1)
                elif request.form["submit_button"] == "live":
                        arr3=live.detectObject()
                        return render_template('search.html',value=arr3)

                        

if __name__ == "__main__":
        app.run(debug=True)
