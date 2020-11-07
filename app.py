from flask import Flask, abort, jsonify, request
from flask_cors import CORS, cross_origin
import time
import json
import datetime
import os
from flask_restful import Resource, Api
import joblib
import inputFile
import numpy as np


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to Group 5 Final Year Project"

@app.route("/api", methods=['POST'])
def make_predict():
    #error checking
    data = request.get_json(force=True)

    #extract url from request
    url_to_be_predicted = data['url']
    url = str(url_to_be_predicted)

    #load the pickle file
    classifier = joblib.load('Models/ML_Pickel/RandomForest.pkl')

    #checking and predicting    
    try:
        checkprediction = inputFile.main(url)
        prediction = int(classifier.predict(checkprediction))
        print(prediction)
        result = {"prediction": prediction}
    except Exception as e:
        #print("Error")
        result = {"prediction": -9999}

    #print(prediction)
    #prediction = prediction + 1
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
