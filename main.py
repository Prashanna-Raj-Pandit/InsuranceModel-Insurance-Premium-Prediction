from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd
import pickle

app = Flask(__name__)

try:
    with open("column_transformer.pkl", "rb") as f:
        ct = pickle.load(f)
    model = tf.keras.models.load_model("insurance_model.h5")
except FileNotFoundError as e:
    raise Exception(f"Missing file: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        input_df = pd.DataFrame(
            [[age, sex, bmi, children, smoker, region]],
            columns=["age", "sex", "bmi", "children", "smoker", "region"]
        )
        user_data_norm = ct.transform(input_df)
        insurance_premium = model.predict(user_data_norm, verbose=0)
        prediction = round(float(insurance_premium[0][0]), 2)
        return render_template('index.html', prediction_text=f"Predicted Insurance Premium: ${prediction}")
    except ValueError:
        return render_template('index.html', prediction_text="Error: Please enter valid numeric values")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)