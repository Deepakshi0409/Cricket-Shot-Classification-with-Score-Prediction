import pandas as pd
from flask import Flask,request, url_for, redirect, render_template
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
#from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, jsonify      
import pickle
import numpy as np
#from extract import exx
import glob
import shutil
from PIL import Image
################
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import joblib
import random
##############


PEOPLE_FOLDER = os.path.join('static', 'people_photo')
# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
def model_predict(img_path,thr):
    """
       model_predict will return the preprocessed image
    """
    
    path=r"D:\NagPersonalProj\CricketPoseDetection\CricketShotIdentification\cricketposeproj\export.pkl"
    learn = load_learner(path)
    img = PILImage.create(img_path)
    #img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    result=pred_class
    print(result,pred_idx)
    thr=float(outputs[pred_idx])
    thr=round(thr,4)
    return pred_idx,thr

@app.route('/')
def home():
   return render_template("home.html")

@app.route("/about")  #click here
def about():
   return render_template("page1.html")

@app.route("/abouts")  #click here
def abouts():
   return render_template("page111.html")
   
@app.route("/aboutt")
def aboutt():
   return render_template("about.html")
  

def hello_world():
   return render_template('page1.html')

@app.route("/submit")
def submit():
   return render_template("hp1.html") 




#For Forged Images below predict function
@app.route('/predict',methods=['POST','GET'])
def predict():
     #calculation of zone done

        
        
    if request.method == 'POST':
        # Get the file from post request
        file_path = request.files['file']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(
            #basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)

        thr=0
        preds = model_predict(file_path,thr)
        print(preds)
        a=preds[0]
        b=preds[1]
        print(a)
        print(type(a))
        print(b)
        b=round(b,2)
        print(b)
        regressor = joblib.load(r'D:\NagPersonalProj\CricketPoseDetection\CricketShotIdentification\cricketposeproj\path\cricket_model.pkl')
        ct = ColumnTransformer([
        ('scale', StandardScaler(), ['match_situation']),
        ('onehot', OneHotEncoder(), ['shot_type', 'shot_zone', 'bowler_type', 'fielder_position'])
        ], remainder='passthrough')
        train_data = pd.read_csv(r'D:\NagPersonalProj\CricketPoseDetection\CricketShotIdentification\cricketposeproj\path\cricket_dataset.csv')
        X_train = train_data.drop('score', axis=1)
        ct.fit(X_train)
        shot_zone = random.choice(['off', 'leg'])
        bowler_type = random.choice(['fast', 'medium', 'spin'])
        fielder_position = random.choice(['deep', 'forward', 'backward'])
        match_situation = random.uniform(0, 1)
        new_data = pd.DataFrame({
           'shot_type': ['sweep'],
           'shot_zone': [shot_zone],
           'bowler_type': [bowler_type],
           'fielder_position': [fielder_position],
           'match_situation': [match_situation]
        })
        X_new = ct.transform(new_data)
        y_pred = regressor.predict(X_new)
        output=new_data['shot_type'][0], y_pred[0]
        output=int(output[1])
        print(output)
        
        if a==0 and b>0.70:
            return render_template('hp1.html',a="Drive ",b=output)
        elif a==1 and b>0.70:
            return render_template('hp1.html',a="Legglance-flick",b=output)
        elif a==2 and b>0.70:
            return render_template('hp1.html',a="Pullshot",b=output)
        elif a==3 and b>0.70:
            return render_template('hp1.html',a="Sweep",b=output)
        else:
            return render_template('hp1.html',a=" Uploaded image is not the correct one, Please upload proper image")
		



    return None

if __name__ == '__main__':
    app.run(debug=True)

