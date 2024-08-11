function checkForPhishing() {
    const fileInput = document.getElementById('emailFile');
    const contentTextarea = document.getElementById('emailContent');
    const resultSection = document.getElementById('result');
    const resultText = document.getElementById('resultText');
    const confidenceScore = document.getElementById('confidenceScore');
    const loadingIndicator = document.getElementById('loading');

    if (!contentTextarea.value.trim()) {
        resultText.textContent = 'Please paste email content.';
        confidenceScore.textContent = '';
        resultSection.style.display = 'block'; // Ensure result section is shown
        return;
    }

    const formData = new FormData();
    formData.append('emailContent', contentTextarea.value);

    // Show loading indicator
    loadingIndicator.style.display = 'block';

    fetch('/check', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultText.textContent = data.isPhishing ? 'Phishing detected!' : 'No phishing detected.';
        confidenceScore.textContent = `Confidence Score: ${data.confidence}`;
        resultSection.style.display = 'block'; // Show result section
    })
    .catch(error => {
        resultText.textContent = 'An error occurred. Please try again.';
        confidenceScore.textContent = '';
        resultSection.style.display = 'block'; // Show result section
    })
    .finally(() => {
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
    });
}

document.getElementById('emailForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData(this);
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/check-phishing', true);

    xhr.onload = function() {
        if (xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            document.getElementById('resultText').textContent = response.result;
            document.getElementById('confidenceScore').textContent = response.confidence;
            document.getElementById('result').style.display = 'block';
        } else {
            document.getElementById('resultText').textContent = 'An error occurred. Please try again.';
            document.getElementById('confidenceScore').textContent = '';
            document.getElementById('result').style.display = 'block';
        }
        document.getElementById('loading').style.display = 'none';
    };

    document.getElementById('loading').style.display = 'block';
    xhr.send(formData);
});
