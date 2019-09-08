from flask import Flask, redirect, url_for, request
app = Flask(__name__)
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import tensorflow as tf
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

global graph,model
graph = tf.get_default_graph()


sc = StandardScaler()

# Initialising the ANN

model = Sequential()

dataset = pd.read_csv('data/dataset2.csv')
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Adding the input layer and the first hidden layer (with dropout)
model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10)) # 10 nodes in hidden layer, initialize weights uniformly, use rectifier funtion for hidden layer, except 11 input nodes
 # Disable 10% of the neurons on each iteration

# Adding the second hidden layer (with dropout)
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) # input_dim already specified in previous hidden layer
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu')) # input_dim already specified in previous hidden layer
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu')) # input_dim already specified in previous hidden layer

model.add(Dense(units = 1, kernel_initializer = 'uniform',activation='linear')) # only 1 node in output layer, use sigmoid function for probability

def avg(y_true, y_pred):
    return k.mean(k.abs(y_pred - y_true))

model.load_weights('airfare.h5')
model.compile(optimizer = 'Adadelta', loss = 'mean_squared_error',metrics=[avg]) # loss defined this way since output is binary, only 1 output node

@app.route('/success/<name>')
def success(name):
   return '<html><b></b><br/> <div id="chartContainer" style="height: 600px; width: 1000px;"><script src="https://canvasjs.com/assets/script/canvasjs.min.js"></script><script>var datay=%s; var datax = [];for (var i = 1; i <= datay.length; i++) {datax.push(i);}  var dps = [];  var chart = new CanvasJS.Chart("chartContainer", { title: { text: "Price Prediction " }, axisX: { title: "Future Days" }, axisY: { title: "Price" }, data: [{ type: "line", dataPoints: dps }] }); function parseDataPoints() { for (var i = dps.length; i < datax.length; i++) dps.push({ x: datax[i], y: datay[i] }); }; function addDataPoints(){ parseDataPoints(); chart.options.data[0].dataPoints = dps; chart.render(); } addDataPoints();  </script></html>'% name

@app.route('/index',methods = ['POST', 'GET'])
def index():
   if request.method == 'POST':
       ct = request.form['ct']
       Dt = request.form['Dt']
       d  = request.form['d']
       a  = request.form['a']
       aw = request.form['aw']
       N  = request.form['N']
       dtf  = int(request.form['dtf'])
       currentday=(datetime.datetime.today().weekday()+1)%7+1
       with graph.as_default():
         
            new_prediction=[]
            for x in range(int(N)):
              prediction=model.predict(sc.transform(np.array([[Dt, ct, aw, a, d ,dtf,(currentday-1+dtf)%7+1,x,(currentday-1+x)%7+1, currentday]])))

              new_prediction.append(prediction[0][0])

       return redirect(url_for('success',name = json.dumps((new_prediction),cls=MyEncoder)) )  
   else:
      #user = request.args.get('nm')
      return redirect(url_for('success',name = "skks"))

if __name__ == '__main__':
   app.run(debug = True)
