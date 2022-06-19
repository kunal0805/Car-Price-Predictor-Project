from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import joblib
app=Flask(__name__)
cors=CORS(app)
# model=pickle.load(open('Model/LinearRegressionModel.pkl'))
import os
from sklearn.preprocessing import OneHotEncoder
location = 'C:/Users/Vivaan/Desktop/CPP/CPP\Model'
fullpath = os.path.join(location, 'LinearRegressionModel.pkl')
model = joblib.load(fullpath)
car=pd.read_csv("C:/Users/Vivaan/Desktop/CPP/CPP/Cleaned_Car_data.csv")

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



# if __name__=='__main__':
#     app.run()

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5000, debug=True)