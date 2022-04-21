# House Price Prediction Group - 177 

from crypt import methods
from flask import Flask , render_template, request
import pandas as pd
import pickle as p 
import numpy as np

app = Flask(__name__)

pricel=[]
for i in range(100,5000,100):
    pricel.append(i)
# xgboost = p.load(open('/Users/prakharraj1302/Desktop/HPP/test_1/xgb_model.plk' , 'rb'))
# model_list = ["XGBoost" , "Support Vector Machine ", "Random Forest Model" , "Keras Neural Network "]
model_list = ["XGBoost" , "Support Vector Machine ", "Random Forest Model" , "Keras Neural Network " , "linear Regression " , "dtree" , "lasso"]

model_dict = {
    1:"XGBoost",
    2:"Support Vector Machine",
    3:"Random Forest Model",
    4:"Keras Neural Network",
    5:"Linear Regression",
    6:"Decision Tree",
    7:"Lasso Regression Model"
}

xgboost = p.load(open('assets/model/xgb_model.pkl' , 'rb'))
svm = p.load(open('assets/model/svm_modle.pkl' , 'rb')) 
rforest = p.load(open('assets/model/random_forest_model.pkl' ,'rb')) 
keras_nn =p.load(open('assets/model/keras_nn_1.pkl' , 'rb'))
linreg =p.load(open('assets/model/linear_reg_model.pkl' , 'rb'))
dtree =p.load(open('assets/model/dtree_model.pkl' , 'rb'))
lasso =p.load(open('assets/model/lasso_model.pkl' , 'rb'))


locations = pd.read_csv('static/location_list.csv')
location = sorted(locations)
location.insert(0 , '')
bhk = ['',1,2,3]
bath = ['',1,2,3]
col = p.load(open('assets/col_data_2.pkl' , 'rb'))

@app.route('/')

def main():

    return render_template('predict.html' ,
    locations = locations ,
    bhk = bhk ,
    bath = bath,
    price = pricel,
    model_dict= model_dict  )


@app.route('/predict' , methods = ['POST'])
def predict():

    loc = request.form.get('loc')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('sqft')
    model =request.form.get('model')

    print(loc , bhk , bath , sqft , model , type(model))

    prediction = 100

    # price = pred_1(xgboost ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    price = selector(model , sqft , bath , bhk , loc)
    if model != "4":
        price = str("₹ {}".format(int(price * 10e4)))
    else:
        price = str("₹ {}".format(int(price)))
    # price = pred_1(xgboost , 1000, 3, 3 , 'Ambegaon Budruk' , 'Super built-up  Area')
    # return str(prediction)
    return price

# def selector(model ,sqft, bath ,bhk , loc):
#     print("_______________________-")
#     Price  = 0
#     if model == "1":
#         price = pred_1(xgboost ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
#     elif model == "2":
#         price = pred_1(svm ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
#     elif model == "3":
#         price = pred_1(rforest ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
#     elif model == "4":
#         price = pred_1(keras_nn ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
#     else:
#         print("ERROR")

#     return price
def selector(model ,sqft, bath ,bhk , loc):
    print("_______________________-")
    Price  = 0
    if model == "1":
        price = pred_1(xgboost ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "2":
        price = pred_1(svm ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "3":
        price = pred_1(rforest ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "4":
        price = pred_1(keras_nn ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "5":
        price = pred_1(linreg ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "6":
        price = pred_1(dtree ,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    elif model == "7":
        price = pred_1(lasso,sqft,bath,  bhk ,loc ,'Super built-up  Area' )
    else:
        print("---------ERROR---------")

    return price

def pred_1 ( model_ ,sqft, bath, bhk ,location ,area_type ):

    loc_index = np.where(col.columns == location)[0][0]
    area_type_index = np.where(col.columns == area_type)[0][0]

    x = np.zeros(len(col.columns))
    x[1] = bath
    x[0] = sqft
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    model =  model_
    if area_type_index >= 0:
        x[area_type_index] = 1

    # print([x])
    dx = pd.DataFrame([x] , columns = col.columns )
    return model.predict(dx)[0]

if __name__ == "__main__" :
    app.run(debug = True)
