from flask import Flask, request, jsonify

app = Flask(__name__)

def check_for_phishing(content):
    # Dummy phishing detection logic (replace with actual implementation)
    if "free money" in content.lower():
        return True, 0.95  # Example confidence score
    return False, 0.1  # Example confidence score

@app.route('/check', methods=['POST'])
def check():
    # Check if request contains file or text
    if 'emailFile' in request.files:
        file = request.files['emailFile']
        # Read file content
        content = file.read().decode('utf-8')
    elif 'emailContent' in request.form:
        content = request.form['emailContent']
    else:
        return jsonify({'error': 'No content provided'}), 400
    
    # Perform phishing check
    is_phishing, confidence = check_for_phishing(content)
    
    # Return result as JSON
    return jsonify({
        'isPhishing': is_phishing,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
