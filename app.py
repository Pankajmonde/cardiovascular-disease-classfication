from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from inference import predict_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_audio(filepath)
            # Clean up the file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
if __name__ == '__main__':
    app.run(debug=True, port=5000)
