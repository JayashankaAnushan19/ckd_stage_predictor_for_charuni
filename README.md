# CKD Stage Predictor

This project is a web-based CKD (Chronic Kidney Disease) Stage Predictor.
It takes patient clinical data and predicts the CKD stage using a pre-trained machine learning model. The frontend provides a visual gauge and bar representation of the kidney function.

## Features
- Interactive web UI to input patient data
- Prediction of CKD stage using pre-trained model
- Visual bar and speedometer for kidney function
- Fully compatible with your best_model.joblib and label_encoder.joblib

## How to Run
1. Install Required Packages
- Open a terminal or command prompt and run:

```
pip install fastapi uvicorn joblib pandas
pip install scikit-learn==1.6.1
```

2. Start the App
- Make sure your folder structure looks like this:
```
CDK_APP/
│   main.py
│   start_app.bat
│   ToRun.md
├───models/
│       best_model.joblib
│       label_encoder.joblib
├───static/
│       index.html

```

- Double-click `start_app.bat`.

3. Open in Browser
- After running the .bat, open your browser and go to:
`http://127.0.0.1:8000`
- Enter patient details in the form and click Predict CKD Stage.
