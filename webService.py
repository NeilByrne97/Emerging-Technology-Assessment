from flask import Flask,redirect,request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

# Declare app
app = Flask(__name__)
# This code is adapted from the Jupyter notebook
data = pd.read_csv('powerproduction.csv')
# Outliers found in Jupyter notebook
data = data.drop([208, 340, 404, 456, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499])

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values
X = X.reshape(-1, 1)

# Decision Tree Regression Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # Training set is 20% of total
decregressor = DecisionTreeRegressor()  # Perfrom the Regression
decregressor.fit(X_train, y_train)

def decTree(speed): # Decision Tree Regression
    speedArr= np.array(speed).reshape(-1, 1)
    return str(decregressor.predict(speedArr)[0])

# Return home 
@app.route('/')
def home():
    return redirect("static/index.html")
speed = 0 # Initialise result

# Decision Tree Regression Training
@app.route("/api/model1", methods = ["GET", "POST"])
def uniform():
    global speed  # Speed is the answer
    if request.method == "POST":
        speed = float(request.json)
    return {"value": decTree(speed)}

# Run
if __name__ == '__main__':
    app.run()