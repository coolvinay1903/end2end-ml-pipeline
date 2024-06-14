import joblib
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd


from custom_transformers import MultiColumnLabelEncoder

app = Flask(__name__)

models = joblib.load("models.pkl")
preprocessing = joblib.load("preprocessing.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    data = [data] if not isinstance(data, list) else data
    print(data)
    df = pd.DataFrame(data)
    print(df)
    df_transformed = preprocessing.transform(df)
    print(df_transformed)
    results = []
    for _, model in models.items():
        results.append(model.predict(df_transformed))
    mean = np.array(results).mean()
    std = np.array(results).std()

    return jsonify({"mean": mean, "std": std})


if __name__ == "__main__":
    app.run(debug=True)
