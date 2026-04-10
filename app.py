from flask import Flask, request, render_template, jsonify
import os
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from inference import predict_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  name TEXT,
                  age TEXT,
                  gender TEXT,
                  symptoms TEXT,
                  filename TEXT,
                  prediction TEXT,
                  confidence REAL,
                  risk_level TEXT,
                  contact TEXT)''')
    try:
        c.execute('ALTER TABLE history ADD COLUMN contact TEXT')
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/analysis')
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
                
            if 'error' in result:
                return jsonify({'error': result['error']})
                
            # Extract patient details
            name = request.form.get('patName', 'Unknown')
            name = name if name else 'Unknown'
            contact = request.form.get('patContact', 'N/A')
            contact = contact if contact else 'N/A'
            age = request.form.get('patAge', 'N/A')
            gender = request.form.get('patGender', 'N/A')
            symptoms = request.form.get('patSymptoms', 'N/A')
            
            # Compute Risk Level
            conf_percent = result['confidence'] * 100
            if result['prediction'] == 'Normal':
                risk_level = 'Low'
            else:
                risk_level = 'Moderate' if conf_percent < 80 else 'High'
                
            result['risk_level'] = risk_level
            
            # Save to Database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute('''INSERT INTO history (date, name, age, gender, symptoms, filename, prediction, confidence, risk_level, contact)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, age, gender, symptoms, file.filename, result['prediction'], conf_percent, risk_level, contact))
            conn.commit()
            conn.close()
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
@app.route('/history/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('DELETE FROM history WHERE id = ?', (record_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return jsonify([dict(ix) for ix in rows])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
