# -*- coding: utf-8 -*-

#importing libraries
import joblib
import inputFile

#load the pickle file
classifier = joblib.load('Models/ML_Pickel/RandomForest.pkl')

#input url
print("enter url")
url = input()

#checking and predicting
checkprediction = inputFile.main(url)
prediction = classifier.predict(checkprediction)

# print(prediction)

# x = prediction.tolist()
#print(type(prediction))

print(prediction)