import flask as fl
import numpy as np

# Create web app
app = fl.Flask(__name__)

# Add root route
@app.route('/')
def stdnor():
    return {"value": np.random.standard_normal()}