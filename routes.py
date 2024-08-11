# app/routes.py

from flask import Blueprint, request, jsonify, render_template
from .utils import check_for_phishing  # Import your phishing check function

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/check-phishing', methods=['POST'])
def check_phishing():
    email_content = request.form['emailContent']
    try:
        result, confidence = check_for_phishing(email_content)  # Process the email content
        return jsonify({
            'result': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'result': 'An error occurred. Please try again.',
            'confidence': ''
        })
