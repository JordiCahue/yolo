from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import base64

import sys

def main_json_predict(image_name) :

   # Open the image file
    with open(image_name, 'rb') as image_file:
        image_data = image_file.read()
    # Convert the image data to a base64-encoded string
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # Serialize the base64-encoded string (for example, send it over a network or save it to a file)
    serialized_data = '{"image":'+'"'+base64_image+'"}'

    print (serialized_data)

def main_json_count(image_name, label) :

    # Open the image file
    with open(image_name, 'rb') as image_file:
        image_data = image_file.read()
    # Convert the image data to a base64-encoded string
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # Serialize the base64-encoded string (for example, send it over a network or save it to a file)
    serialized_data = '{'+'"label":'+'"'+label+'", "image":'+'"'+base64_image+'"}'

    print (serialized_data)

def main_predict(image_name, model_name) :

    # Open the image file
    with open(image_name, 'rb') as image_file:
        image_data = image_file.read()

    pil_image = Image.open(BytesIO(image_data))
    image = np.asarray(pil_image)

    # Cargamos el modelo
    model = YOLO(model_name)

    print("Using model "+model_name)
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

    print (output)


# mytool op='predict' image='./image/abc.jpeg'
# mytool op='json-predict' image='./image/abc.jpeg'
# mytool op='json-count' label='label' image='./image/abc.jpeg'


if __name__ == '__main__':

    error = False
    if (len(sys.argv) == 3 or len(sys.argv) == 4) :

        if (len(sys.argv) >= 3) :

            param1 = sys.argv[1]

            p1 = param1.split('=')[0]
            v1 = param1.split('=')[1]

            param2 = sys.argv[2]

            p2 = param2.split('=')[0]
            v2 = param2.split('=')[1]

            p3 = None
            if (len(sys.argv)>= 4) :
                param3 = sys.argv[3]

                p3 = param3.split('=')[0]
                v3 = param3.split('=')[1]

            if (p1 == 'op') : op = v1
            if (p2 == 'image') : image = v2
            if (p3 and p3 == 'label') : label = v3
            if (p3 and p3 == 'model') : model = v3

            if (op and op == 'json-predict' and image) :
                main_json_predict(image)
            elif (op and op == 'json-count' and label) :
                main_json_count(image, label)
            elif (op and op == 'predict' and model) :
                main_predict(image, model)
            else : error = True

    else :
        error = True
    
    if (error) :
        print ("Usage:")
        print ("    mytool op='predict' image='./image/abc.jpeg' model='yolov8m.pt")
        print ("    mytool op='json-predict' image='./image/abc.jpeg'")
        print ("    mytool op='json-count' image='./image/abc.jpeg' label='label'")



    




