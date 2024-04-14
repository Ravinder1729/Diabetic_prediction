from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    # Retrieve form data
    Pregnancies = request.form.get("Pregnancies")
    Glucose = request.form.get("Glucose")
    BloodPressure = request.form.get("BloodPressure")
    SkinThickness = request.form.get("SkinThickness")
    Insulin = request.form.get("Insulin")
    BMI = request.form.get("BMI")
    DiabetesPedigreeFunction = request.form.get("DiabetesPedigreeFunction")
    Age = request.form.get("Age")

    
    try:
        Pregnancies = float(Pregnancies) if Pregnancies is not None else None
        Glucose = float(Glucose) if Glucose is not None else None
        BloodPressure = float(BloodPressure) if BloodPressure is not None else None
        SkinThickness = float(SkinThickness) if SkinThickness is not None else None
        Insulin = float(Insulin) if Insulin is not None else None
        BMI = float(BMI) if BMI is not None else None
        DiabetesPedigreeFunction = float(DiabetesPedigreeFunction) if DiabetesPedigreeFunction is not None else None
        Age = float(Age) if Age is not None else None
    except ValueError:
        return "Error: Invalid input. Please enter valid numbers for all fields"

    # Load model
    with open(r"D:\diabetic\daibetic2.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

    # Convert the input data to a numpy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=np.float32)

    # Perform prediction
    prediction = model.predict(input_data)[0]

    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
