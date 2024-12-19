from flask import Flask, redirect, session, url_for, render_template, request, jsonify
import json
import random
import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_session import Session
from datetime import datetime
from collections import Counter

locations = [
    [8.699900284031973, 77.74306222856937],
    [8.70528778413457, 77.75456354073003],
    [8.711056989576877, 77.74018690052918],
    [8.710887308333989, 77.73224756191081],
    [8.693070349647227, 77.74254724444276]
]


selected_locations = random.sample(locations, 1)


now = datetime.now()
current_date = now.date()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'your_secret_key' 
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Define JSON data structure
json_data = {
    "date": "",
    "location": ""
}
dic = []
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/view_metrics')
def view_metrics():
    return render_template('view_performance.html')

@app.route('/index')
def index():
    session['mode'] = ""
    return render_template('index.html')

@app.route('/predict')
def predict():
    session['mode'] =""
    return render_template('predict.html')

def store_data(new_data):
    json_file_path = os.path.join("static/", 'prediction_results.json')
    
    # Read existing data
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = []  
            except json.JSONDecodeError:
                existing_data = []  
    else:
        existing_data = []  
    
    # Append new data
    if isinstance(existing_data, list):
        existing_data.append(new_data)
    else:
        existing_data = [new_data]
    
    # Write updated data
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)  

def load_and_sort_data():
    with open('static/prediction_results.json', 'r') as file:
        data = json.load(file)

    for item in data:
        item["date"] = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
    

    data.sort(key=lambda x: x["date"])
    
    return data


def Test(video_path, result_folder):
    print(f"Processing video: {video_path}")
    save_model = keras.models.load_model('F:/2024/sarah_2024/Projects/Fire_detection/my_keras_model.keras')
    class_names = ['fire_images', 'non_fire_images']
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    final_res = ""
    while frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frames are left

        frame_count += 1
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        
        predict = save_model.predict(image)
        pred = np.where(predict >= 0.5, 1, 0)
        
        dic.append(class_names[pred[0][0]])
        final_res = class_names[pred[0][0]]
        
        result_path = os.path.join(result_folder, f'frame_{frame_count}.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        plt.title(f'Frame {frame_count}: {class_names[pred[0][0]]}')
        session['predictions'] = class_names[pred[0][0]]
        print(f"predicted:{class_names[pred[0][0]]}")
        plt.axis("off")
        plt.savefig(result_path)
        plt.close()
        
    cap.release()
    cv2.destroyAllWindows()
    label_counts = Counter(dic)
    most_common_label, highest_count = label_counts.most_common(1)[0]
    session['mode'] = most_common_label
    print(f"The label with the highest count is '{most_common_label}' with {highest_count} occurrences.")
    print(f"*************************{dic}")
    
    if most_common_label == "fire_images":
        json_data = {
            "date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "location": selected_locations
            }
            
        store_data(json_data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    session['predictions'] = None
    if request.method == 'POST':
        if 'videoFile' not in request.files:
            return redirect(request.url)
        file = request.files['videoFile']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            # Call Test function
            Test(video_path, app.config['RESULT_FOLDER'])
            
            return redirect(url_for('result'))
    return render_template('predict.html')

@app.route('/result')
def result():
    result_files = os.listdir(app.config['RESULT_FOLDER'])
    result_files = sorted(result_files)
    return render_template('predict.html', files=result_files)

@app.route('/login', methods=['GET', 'POST'])
def login(): 
    return render_template('login.html')

@app.route('/authorised', methods=['GET', 'POST'])
def authorised(): 
    email = request.form.get("email")
    pwd = request.form.get("pwd")
    if email == "admin@gmail.com" and pwd == "admin":
        data = load_and_sort_data()
        print(data)
        return render_template('admin.html' , data=data)
    else:
        return render_template('login.html', message="Sorry Invalid credentials")


@app.route('/map_view/<float:lat>/<float:lng>')
def map_view(lat, lng):
    return render_template('map.html', lat=lat, lng=lng)


if __name__ == '__main__':
    app.run(port=5006)
