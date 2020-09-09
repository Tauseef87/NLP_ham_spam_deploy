import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('nlp_model.pkl', 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
#The below route will render index.html 
@app.route('/')
def home():
    return render_template('home.html')

#The below route will work for predict button
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_predict = model.predict(vect)
    return render_template('Result.html', prediction=my_predict)

#The below part run the whole flask
if __name__ == "__main__":
    app.run(debug=True)