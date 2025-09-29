from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# --- Carichiamo modello e scaler ---
rf_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

app = Flask(__name__)

# --- Endpoint frontend ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Endpoint API per predizione ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Creiamo DataFrame
    df = pd.DataFrame([[
        data['Age'], data['Gender'], data['Grade_Level'],
        data['Strength_Score'], data['Endurance_Score'], data['Flexibility_Score'],
        data['Speed_Agility_Score'], data['BMI'], data['Health_Fitness_Knowledge_Score'],
        data['Skills_Score'], data['Class_Participation_Level'], data['Attendance_Rate'],
        data['Motivation_Level'], data['Overall_PE_Performance_Score'], data['Improvement_Rate'],
        data['Final_Grade'], data['Previous_Semester_PE_Grade'], data['Hours_Physical_Activity_Per_Week']
    ]], columns=[
        "Age","Gender","Grade_Level","Strength_Score","Endurance_Score","Flexibility_Score",
        "Speed_Agility_Score","BMI","Health_Fitness_Knowledge_Score","Skills_Score",
        "Class_Participation_Level","Attendance_Rate","Motivation_Level","Overall_PE_Performance_Score",
        "Improvement_Rate","Final_Grade","Previous_Semester_PE_Grade","Hours_Physical_Activity_Per_Week"
    ])
    
    # Trasformazione colonne categoriche
    for col in df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Standardizziamo
    df_scaled = scaler.transform(df)
    
    # Predizione
    pred = rf_model.predict(df_scaled)
    pred_label = label_encoders["Performance"].inverse_transform(pred)[0]
    
    return jsonify({"performance": pred_label})

if __name__ == "__main__":
    app.run(debug=True)
