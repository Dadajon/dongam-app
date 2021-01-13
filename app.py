from __future__ import division, print_function

import os
import warnings

import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from datetime import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)

# Define a flask app
app = Flask(__name__)

model_path = "models/whole-69/model-best.h5"

# Load trained model
model = load_model(model_path, compile=False)
model._make_predict_function()  # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', f"{dt_string}_{secure_filename(f.filename)}")
        f.save(file_path)

        # class 69 labels
        labels = ['칡', '결명자', '고본', '당근', '회향', '배초향', '방아풀', '산박하', '구기자', '부추', '파(실파)', '금불초', '망강남', '도라지', '백도라지', '잔대(층층잔대)', '더덕', '독활', '두릅나무', '쇠비름', '비름나물', '으름덩굴', '통탈목', '멀꿀', '구릿대', '궁궁이', '백출(삽주)', '당백출(큰꽃삽주)', '산수유', '산딸나무', '생각나무', '갯기름나물', '갯방풍','중국방풍', '약모밀', '삼백초', '오미자', '흑오미자', '남오미자', '우방자=우엉', '환삼덩굴', '호프', '익모초', '인동', '들깨', '참깨', '지황', '곰보배추', '천궁', '토천궁', '치자나무', '항유', '꽃향유', '좀향유', '가는잎향유', '현삼', '큰개현삼', '토현삼', '섬현삼', '호도', '목적(속새)', '쇠뜨기', '황금', '골무꽃', '황기', '고삼', '사철쑥', '더위지기', '차조기']

        img = image.img_to_array(image.load_img(
            file_path, target_size=(224, 224))) / 255.

        x = img.astype('float16')
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x)

        readable_accuracy = format_pred_acc(predictions[0])
        print(readable_accuracy)
        sort_tuple = tuple(zip(labels, readable_accuracy))
        sorted_val = sorted(sort_tuple, key=lambda x: x[1])
        predictions = predictions.argmax(axis=1)

        if float(sorted_val[-1][1]) == 1.0 and float(sorted_val[-2][1]) < 0.1:
            result = f"Result: {sorted_val[-1][0]} "
        elif float(sorted_val[-1][1]) < 0.991:
            result = "다음의 경우에 해당됩니다. \n 1. 사진을 멀리 찍으셨습니다. \n 2. 학습한 식물이 아닌 것 같습니다."
        else:
            result = f"Top-3 predictions: {sorted_val[-1][0]}   |   {sorted_val[-2][0]}   |   {sorted_val[-3][0]}"
        print(result)
        return result
    return None


def format_pred_acc(acc_array):
    float_formatter = "{:.5f}".format
    readable_accuracy = list()
    for i in acc_array:
        readable_accuracy.append(float_formatter(i))
    return readable_accuracy


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
