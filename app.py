import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## import ridge regresor model and standard scaler pickle
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        symboling = float(request.form.get('symboling'))
        fueltype = float(request.form.get('fueltype'))
        aspiration = float(request.form.get('aspiration'))
        enginelocation = float(request.form.get('enginelocation'))
        wheelbase = float(request.form.get('wheelbase'))
        #carlength = float(request.form.get('carlength'))
        carwidth = float(request.form.get('carwidth'))
        carheight = float(request.form.get('carheight'))
        enginesize = float(request.form.get('enginesize'))
        boreratio = float(request.form.get('boreratio'))
        stroke = float(request.form.get('stroke'))
        #compressionratio = float(request.form.get('compressionratio'))
        horsepower = float(request.form.get('horsepower'))
        peakrpm = float(request.form.get('peakrpm'))
        citympg = float(request.form.get('citympg'))
        #highwaympg = float(request.form.get('highwaympg'))


        new_data_scaled = standard_scaler.transform([[symboling,fueltype,aspiration,enginelocation,wheelbase,carwidth,carheight,enginesize,boreratio,stroke,horsepower,peakrpm,citympg]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',result = result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
