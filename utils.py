# app/utils.py

import joblib
import os

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def check_for_phishing(email_content):
    # Transform the email content using the vectorizer
    processed_content = vectorizer.transform([email_content])
    
    # Make prediction using the loaded model
    prediction = model.predict(processed_content)
    
    # Get confidence score
    confidence = max(model.predict_proba(processed_content)[0])
    
    # Convert numerical prediction to meaningful label
    result = 'Phishing' if prediction[0] == 1 else 'Not Phishing'
    
    return result, confidence
