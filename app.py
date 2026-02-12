import pickle
import json
 
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app= Flask(__name__)
## Load the model
regmodel= pickle.load(open('regmodel.pkl','rb'))
scalar= pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.get_json()
    print("Received JSON:",data)
    input_data = print(np.array(list(data.values())).reshape(1,-1))
    scaled_data  =scalar.transform(input_data)
    output = regmodel.predict(scaled_data )
    print(output[0])
    return jsonify({"prediction":output[0]})



if __name__ == "__main__":
    app.run(debug=True)
