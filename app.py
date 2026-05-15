from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        income = float(request.form['income'])
        age = float(request.form['age'])
        spending = float(request.form['spending'])
        children = float(request.form['children'])
        recency = float(request.form['recency'])

        if income < 0 or age < 0 or spending < 0 or children < 0 or recency < 0:

            error = "Negative values are not allowed."

            return render_template(
                'index.html',
                error=error
            )

        data = np.array([
            [income, age, spending, children, recency]
        ])

        scaled_data = scaler.transform(data)

        cluster = model.predict(scaled_data)[0]

        if cluster == 0:

            segment = "Premium Customers"

            insight = "These customers have high income and high spending behavior."

            recommendation = "Offer premium memberships, luxury products, and exclusive discounts."

        elif cluster == 1:

            segment = "Budget Customers"

            insight = "These customers spend less and are price sensitive."

            recommendation = "Provide discounts, coupons, and budget-friendly offers."

        elif cluster == 2:

            segment = "Regular Customers"

            insight = "These customers have moderate income and stable purchasing behavior."

            recommendation = "Engage them with loyalty programs and seasonal offers."

        else:

            segment = "Potential Customers"

            insight = "These customers have high potential but lower engagement."

            recommendation = "Target them using personalized marketing campaigns."

        prediction_data = {
            'Income': [income],
            'Age': [age],
            'Spending': [spending],
            'Children': [children],
            'Recency': [recency],
            'Cluster': [cluster],
            'Segment': [segment]
        }

        new_data = pd.DataFrame(prediction_data)

        excel_file = 'predictions.xlsx'

        if os.path.exists(excel_file):

            existing_data = pd.read_excel(excel_file)

            updated_data = pd.concat(
                [existing_data, new_data],
                ignore_index=True
            )

            updated_data.to_excel(excel_file, index=False)

        else:

            new_data.to_excel(excel_file, index=False)

        return render_template(
            'result.html',
            cluster=cluster,
            segment=segment,
            insight=insight,
            recommendation=recommendation
        )

    except Exception as e:

        error = "Invalid input. Please enter valid numeric values."

        return render_template(
            'index.html',
            error=error
        )

@app.route('/history')
def history():

    try:

        data = pd.read_excel('predictions.xlsx')

        tables = data.to_html(
            classes='table table-bordered table-striped',
            index=False
        )

        return render_template(
            'history.html',
            tables=tables
        )

    except:

        return "No prediction history found."
if __name__ == '__main__':
    app.run(debug=True)