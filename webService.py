from flask import Flask,redirect,request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

# This code is adapted from the Jupyter notebook
lin_data = pd.read_csv('powerproduction.csv')

# Outliers found in Jupyter notebook
lin_data = lin_data.drop([208, 340, 404, 456, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499])

X = lin_data.iloc[:, 0].values
y = lin_data.iloc[:, 1].values
X = X.reshape(-1, 1)

# Decision Tree Regression Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # Training set is 20% of total
decregressor = DecisionTreeRegressor()  # Perfrom the Regression
decregressor.fit(X_train, y_train)

def decTree(speed):
    speed_arr = np.array(speed).reshape(-1, 1)
    return str(decregressor.predict(speed_arr)[0])

# declare app
app = Flask(__name__)

# Return home 
@app.route('/')
def home():
    return redirect("static/index.html")
speed = 0

# Decision Tree Regression Training
@app.route("/api/model1", methods = ["GET", "POST"])
def uniform():
    global speed
    if request.method == "POST":
        speed = float(request.json)
    return {"value": decTree(speed)}

# Run
if __name__ == '__main__':
    app.run()