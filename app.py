
from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "titanic_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get passenger details from HTML form
        pclass = int(request.form["pclass"])
        sex = request.form["sex"]
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = request.form["embarked"]

        # Encode categorical variables
        # Same alphabetical encoding used by LabelEncoder
        sex_encoded = {
            "female": 0,
            "male": 1
        }[sex]

        embarked_encoded = {
            "C": 0,
            "Q": 1,
            "S": 2
        }[embarked]

        # Prepare input for the model
        input_data = pd.DataFrame(
            [[
                pclass,
                sex_encoded,
                age,
                sibsp,
                parch,
                fare,
                embarked_encoded
            ]],
            columns=[
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked"
            ]
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get survival probability
        probabilities = model.predict_proba(input_data)[0]

        survival_probability = (
            probabilities[list(model.classes_).index(1)] * 100
        )

        # Prepare result
        if prediction == 1:
            result = "Survived"
            message = "The model predicts that this passenger survived!"
        else:
            result = "Did Not Survive"
            message = "The model predicts that this passenger did not survive."

        return render_template(
            "index.html",
            result=result,
            message=message,
            probability=round(survival_probability, 2)
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid passenger information."
        )


if __name__ == "__main__":
    app.run(debug=True)

