import flask
import pickle
import pandas as pd
from flask import url_for

app = flask.Flask(__name__, template_folder='templates')

with open('framingham_classifier_Logistic_regression_new.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        age = flask.request.form['age']
        sysBP = flask.request.form['sysBP']
        diaBP = flask.request.form['diaBP']
        glucose = flask.request.form['glucose']
        diabetes = flask.request.form['diabetes']
        male = flask.request.form['male']
        BPMeds = flask.request.form['BPMeds']
        totChol = flask.request.form['totChol']
        BMI = flask.request.form['BMI']
        prevalentStroke = flask.request.form['prevalentStroke']
        prevalentHyp = flask.request.form['prevalentHyp']
        input_variables = pd.DataFrame(
            [[age, sysBP, diaBP, glucose, diabetes, male, BPMeds, totChol, BMI, prevalentStroke, prevalentHyp]],
            columns=['age', 'sysBP', 'diaBP', 'glucose', 'diabetes', 'male', 'BPMeds', 'totChol', 'BMI',
                     'prevalentStroke', 'prevalentHyp'],
            dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Age': age,
                                                     'Systolic BP': sysBP,
                                                     'Diastolic BP': diaBP,
                                                     'Glucose': glucose,
                                                     'Diabetes': diabetes,
                                                     'Gender': male,
                                                     'BP Medication': BPMeds,
                                                     'Total Cholesterol': totChol,
                                                     'BMI': BMI,
                                                     'Prevalent Stroke': prevalentStroke,
                                                     'Prevalent Hypertension': prevalentHyp},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
