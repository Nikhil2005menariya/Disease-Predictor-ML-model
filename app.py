from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('model/diseasemodel.pkl')  # replace with your model's filename

# Load the dataset to get the symptoms
dt = pd.read_csv("Training.csv").loc[:, ~pd.read_csv("Training.csv").columns.str.contains('^Unnamed')]
symptom_list = dt.drop(columns=["prognosis"]).columns.tolist()
symptom_list.sort()

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get symptoms from the form (selected symptoms)
        symptoms_input = request.form.getlist('symptoms')  # List of symptoms selected

        # Create a binary feature vector (1 for selected, 0 for not selected)
        input_features = np.zeros(len(symptom_list))
        
        for symptom in symptoms_input:
            if symptom in symptom_list:
                index = symptom_list.index(symptom)
                input_features[index] = 1  # Mark the symptom as present

        # Reshape to match the model's expected input shape
        input_features = input_features.reshape(1, -1)

        # Predict the disease
        prediction = model.predict(input_features)

        # Return the result to the user
        return render_template('index.html', prediction=prediction[0], symptoms=symptom_list)

if __name__ == '__main__':
    app.run(debug=True)
