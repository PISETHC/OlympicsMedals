from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_url_path='/static')

# Load your dataset (Update the file path as needed)
x = pd.read_csv("teams.csv")
x = x[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
x[['athletes', 'prev_medals']] = imputer.fit_transform(x[['athletes', 'prev_medals']])

# Train a linear regression model
train = x[x["year"] < 2012].copy()
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        athletes = float(request.form['athletes'])
        prev_medals = float(request.form['prev_medals'])

        # Make predictions using the trained model
        prediction = reg.predict([[athletes, prev_medals]])
        prediction = max(0, prediction[0])  # Ensure predictions are non-negative

        # Pass the prediction to the template
        return render_template('result.html', athletes=athletes, prev_medals=prev_medals, prediction=prediction)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

# Add a new route for the start page
@app.route('/start')
def start():
    return render_template('start.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
