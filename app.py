from __future__ import division, print_function

import os
import warnings
import logging
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import eli5
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


warnings.simplefilter("ignore")  # disable Keras warnings

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
        labels = ['칡', '결명자', '고본', '당근', '회향', '배초향', '방아풀', '산박하', '구기자', '부추', '파(실파)', '금불초', '망강남', '도라지', '백도라지', '잔대(층층잔대)', '더덕', '독활', '두릅나무', '쇠비름', '비름나물', '으름덩굴', '통탈목', '멀꿀', '구릿대', '궁궁이', '백출(삽주)', '당백출(큰꽃삽주)', '산수유', '산딸나무', '생각나무', '갯기름나물', '갯방풍','중국방풍', '약모밀', '삼백초', '오미자', '흑오미자', '남오미자', '우방자=우엉', '환삼덩굴', '호프', '익모초', '인동', '들깨', '참깨', '지황', '곰보배추', '천궁', '토천궁', '치자나무', '항유', '꽃향유', '좀향유', '가는잎향유', '현삼', '큰개현삼', '토현삼', '섬현삼', '호도', '목적(속새)', '쇠뜨기', '황금', '골무꽃', '황기', '고삼', '사철쑥', '더위지기', '차즈기']

        img = img_to_array(load_img(file_path, target_size=(224, 224))) / 255.
        x = np.expand_dims(img, axis=0)

        datagen = ImageDataGenerator(
            zoom_range=[0.4, 1.0], width_shift_range=[-30, 30], height_shift_range=[-30, 30])

        it = datagen.flow(x, batch_size=1)

        predictions = model.predict(x)

        readable_accuracy = format_pred_acc(predictions[0])
        print(readable_accuracy)
        sort_tuple = tuple(zip(labels, readable_accuracy))
        sorted_val = sorted(sort_tuple, key=lambda x: x[1])
        predictions = predictions.argmax(axis=1)

        if float(sorted_val[-1][1]) == 1.0 and float(sorted_val[-2][1]) < 0.1:
            result = f"{sorted_val[-1]}"
        elif 0.991 < float(sorted_val[-1][1]) < 1.0:
            in2model_img = list()
            for i in range(9):
                batch = it.next()
                image = batch[0]
                in2model_img.append(image)
            most_alike = list()
            for i in range(9):
                input_to_model = np.expand_dims(in2model_img[i], axis=0)
                preds = model.predict(input_to_model)
                readable_acc = format_pred_acc(preds[0])
                if float(max(readable_acc)) > 0.5:
                    sort_tuple = tuple(zip(labels, readable_acc))
                    sorted_val = sorted(sort_tuple, key=lambda x: x[1])
                    most_alike.append(sorted_val[-1])
                    most_alike.append(sorted_val[-2])
                    most_alike.append(sorted_val[-3])
            pred_freq_arr = count_frequency(most_alike)
            # calculate weighted average. Failed
            sorted_freq = sorted(pred_freq_arr.items(), key=lambda item: item[1])
            result = f"{sorted_freq[-1]}, {sorted_freq[-2]}, {sorted_freq[-3]}"
        else:
            result = "다음의 경우에 해당됩니다. \n 1. 사진을 멀리 찍으셨습니다. \n 2. 학습한 식물이 아닌 것 같습니다."
        print(result)

        return result
    return None


def format_pred_acc(acc_array):
    float_formatter = "{:.5f}".format
    readable_accuracy = list()
    for i in acc_array:
        readable_accuracy.append(float_formatter(i))
    return readable_accuracy


def count_frequency(my_list):
    freq = {}
    for item in my_list:
        if item[0] in freq:
            freq[item[0]] += float(item[1])
        else:
            freq[item[0]] = float(item[1])
    for key, value in freq.items():
        print("% s : % f" % (key, value))
    return freq


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
