from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

MERCHANTS = sorted(encoders['merchant'].classes_)
CATEGORIES = sorted(encoders['category'].classes_)
STATES = sorted(encoders['state'].classes_)

stats = {"total": 0, "fraud": 0, "safe": 0}

@app.route('/')
def home():
    return render_template("index.html", merchants=MERCHANTS, categories=CATEGORIES, states=STATES, stats=stats)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        input_data = {
            'merchant': request.form['merchant'],
            'category': request.form['category'],
            'amt': float(request.form['amt']),
            'gender': request.form['gender'],
            'city': request.form['city'],
            'state': request.form['state'],
            'job': request.form['job'],
            'zip': int(request.form['zip']),
            'city_pop': int(request.form['city_pop']),
            'age': int(request.form['age']),
            'hour': int(request.form['hour']),
            'day': int(request.form['day']),
            'weekday': int(request.form['weekday']),
        }

        df = pd.DataFrame([input_data])
        for col, le in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        for col in columns:
            if col not in df.columns:
                df[col] = -999
        df = df[columns]

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        stats["total"] += 1
        if prediction == 1:
            stats["fraud"] += 1
            result = " Fraud Detected!"
        else:
            stats["safe"] += 1
            result = " Safe Transaction"

        return render_template("result.html", result=result, probability=round(prob * 100, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
