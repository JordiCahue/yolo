from flask import Flask, jsonify, request

from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import base64

def start_server():
    model = YOLO("yolov8s.pt")
    print("Model yolov8s.pt loaded")
    print("Backend ready to receive requests")
    return model

app = Flask(__name__)

model = start_server()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    image_data = base64.b64decode(data.get('image'))
    pil_image = Image.open(BytesIO(image_data))
    image = np.asarray(pil_image)

    # Hacemos la prediccion
    results = model.predict(image)
  
    # Número total de objetos detectados
    print(f"Total objects detected: {len(results[0].boxes.boxes)}")

    # serializamos los objetos (score y clase) en un array json
    output = "["
    for result in results:     
        for i, s in enumerate(result.boxes.conf):
            output +=  '{'
            output += '"score"'+': '+str(s.numpy())+', '
            output += '"class"'+': '+'"'+model.names[int(result.boxes.cls[i])]+'"'
            print(model.names[int(result.boxes.cls[i])])
            if (i+1 == len(result.boxes.conf)) :
                output = output + '}'
            else :
                output = output + '}, '
    output += "]"

    # serializamos documento resultados
    output = '{ "results"'+': '+output+'}'

    print (output);

    return output

@app.route('/count', methods=['POST'])
def count():
    data = request.get_json()

    image_data = base64.b64decode(data.get('image'))
    pil_image = Image.open(BytesIO(image_data))
    image = np.asarray(pil_image)

    label = data.get('label')

    # Hacemos la prediccion
    results = model.predict(image)
  
    # Número total de objetos detectados
    print(f"Total objects detected: {len(results[0].boxes.boxes)}")

    # iteramos resultados y contamos cuantos objetos tipo label de han detectado
    count = 0
    for result in results:     
        for i, s in enumerate(result.boxes.conf):
            score = s.numpy()
            object = model.names[int(result.boxes.cls[i])]
            if (object == label) :
                count += 1

    # serializamos documento resultados
    output = '{ "label"'+': '+'"'+label+'", "count"'+': '+str(count)+'}'

    print (output);

    return output

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0') 
    

