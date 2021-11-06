from flask import Flask,render_template,request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

model = load_model('./static/modelMobileNet.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (180,180))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 180,180,3)
    prediction = model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True) #debug=True



