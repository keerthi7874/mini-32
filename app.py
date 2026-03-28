# """
# app.py
# ======
# This is the FLASK WEB SERVER — the heart of the project.

# Simple Explanation:
# -------------------
# Flask is like a waiter in a restaurant:
#   - Customer (browser) sends a request: "Check this URL for me!"
#   - Waiter (Flask) takes the request to the kitchen (ML model)
#   - Kitchen (model) processes it and returns: "PHISHING!" or "SAFE!"
#   - Waiter (Flask) serves the result back to the customer (browser)

# How to Run:
# -----------
#     python app.py
    
# Then open your browser and go to:  http://127.0.0.1:5000
# """

# import os
# import pickle
# import json
# from flask import Flask, render_template, request, jsonify
# from feature_extractor import extract_features, FEATURE_NAMES

# # ──────────────────────────────────────────────────────────────────────────────
# # INITIALIZE FLASK APP
# # ──────────────────────────────────────────────────────────────────────────────

# app = Flask(__name__)

# # ──────────────────────────────────────────────────────────────────────────────
# # LOAD THE TRAINED MODEL
# # ──────────────────────────────────────────────────────────────────────────────

# MODEL_PATH = "model/phishing_model.pkl"
# model_data = None

# def load_model():
#     """Load the saved ML model from disk."""
#     global model_data
#     if os.path.exists(MODEL_PATH):
#         with open(MODEL_PATH, "rb") as f:
#             model_data = pickle.load(f)
#         print(f"✅ Model loaded! Accuracy: {model_data['accuracy']*100:.2f}%")
#     else:
#         print("❌ Model not found! Please run: python train_model.py first")

# load_model()

# # ──────────────────────────────────────────────────────────────────────────────
# # ROUTES (Pages / Endpoints)
# # ──────────────────────────────────────────────────────────────────────────────

# @app.route('/')
# def home():
#     """
#     Home page — shows the URL input form.
#     Flask will look for 'templates/index.html' and serve it.
#     """
#     model_accuracy = None
#     if model_data:
#         model_accuracy = round(model_data['accuracy'] * 100, 2)
#     return render_template('index.html', model_accuracy=model_accuracy)


# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     This route handles the URL check request.
    
#     Flow:
#     1. Get URL from form/JSON
#     2. Extract features
#     3. Ask model: phishing or safe?
#     4. Return result as JSON
#     """

#     # Check if model is loaded
#     if model_data is None:
#         return jsonify({
#             'error': 'Model not loaded. Please run train_model.py first.'
#         }), 500

#     # Get the URL from request
#     data = request.get_json()
#     url = data.get('url', '').strip()

#     if not url:
#         return jsonify({'error': 'Please enter a URL'}), 400

#     # Add http:// if missing (so feature extraction works)
#     if not url.startswith(('http://', 'https://')):
#         url_to_analyze = 'http://' + url
#     else:
#         url_to_analyze = url

#     try:
#         # Extract features from the URL
#         features = extract_features(url_to_analyze)

#         # Get the ML model
#         model = model_data['model']

#         # Predict: 0 = Safe, 1 = Phishing
#         prediction = model.predict([features])[0]

#         # Get probability scores (how confident is the model?)
#         probabilities = model.predict_proba([features])[0]
#         safe_prob = round(float(probabilities[0]) * 100, 2)
#         phish_prob = round(float(probabilities[1]) * 100, 2)

#         # Prepare feature details for display
#         feature_details = []
#         for name, value in zip(FEATURE_NAMES, features):
#             feature_details.append({
#                 'name': name.replace('_', ' ').title(),
#                 'value': round(float(value), 4),
#                 'suspicious': is_suspicious(name, value)
#             })

#         # Determine result
#         is_phishing = bool(prediction == 1)

#         # Risk level
#         if phish_prob >= 80:
#             risk_level = "HIGH RISK"
#             risk_color = "red"
#         elif phish_prob >= 50:
#             risk_level = "MEDIUM RISK"
#             risk_color = "orange"
#         elif phish_prob >= 30:
#             risk_level = "LOW RISK"
#             risk_color = "yellow"
#         else:
#             risk_level = "SAFE"
#             risk_color = "green"

#         return jsonify({
#             'url': url,
#             'is_phishing': is_phishing,
#             'prediction': 'PHISHING' if is_phishing else 'SAFE',
#             'safe_probability': safe_prob,
#             'phishing_probability': phish_prob,
#             'risk_level': risk_level,
#             'risk_color': risk_color,
#             'features': feature_details,
#             'model_accuracy': round(model_data['accuracy'] * 100, 2)
#         })

#     except Exception as e:
#         return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# def is_suspicious(feature_name, value):
#     """
#     Returns True if this feature value looks suspicious.
#     Used for color-coding the feature table in the UI.
#     """
#     suspicious_rules = {
#         'url_length': value > 75,
#         'count_dots': value > 4,
#         'count_hyphens': value > 3,
#         'count_at': value > 0,
#         'has_ip_address': value == 1,
#         'num_subdomains': value > 3,
#         'count_percent': value > 3,
#         'url_entropy': value > 4.5,
#         'has_login': value == 1,
#         'has_secure': value == 1,
#         'has_verify': value == 1,
#         'has_update': value == 1,
#     }
#     return suspicious_rules.get(feature_name, False)


# @app.route('/batch', methods=['POST'])
# def batch_predict():
#     """
#     Check multiple URLs at once.
#     Useful for scanning a list of URLs.
#     """
#     if model_data is None:
#         return jsonify({'error': 'Model not loaded'}), 500

#     data = request.get_json()
#     urls = data.get('urls', [])

#     if not urls or len(urls) > 50:
#         return jsonify({'error': 'Provide 1–50 URLs'}), 400

#     results = []
#     model = model_data['model']

#     for url in urls:
#         url = str(url).strip()
#         if not url.startswith(('http://', 'https://')):
#             url_to_analyze = 'http://' + url
#         else:
#             url_to_analyze = url

#         try:
#             features = extract_features(url_to_analyze)
#             pred = model.predict([features])[0]
#             probs = model.predict_proba([features])[0]
#             results.append({
#                 'url': url,
#                 'is_phishing': bool(pred == 1),
#                 'phishing_probability': round(float(probs[1]) * 100, 2)
#             })
#         except Exception as e:
#             results.append({
#                 'url': url,
#                 'error': str(e)
#             })

#     return jsonify({'results': results})


# @app.route('/api/info')
# def api_info():
#     """Simple endpoint to check if server is running."""
#     return jsonify({
#         'status': 'running',
#         'model_loaded': model_data is not None,
#         'model_accuracy': round(model_data['accuracy'] * 100, 2) if model_data else None,
#         'features_count': len(FEATURE_NAMES),
#         'version': '1.0.0'
#     })


# # ──────────────────────────────────────────────────────────────────────────────
# # RUN THE SERVER
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     print("=" * 60)
#     print("  🌐 PHISHING DETECTOR — FLASK SERVER")
#     print("=" * 60)
#     print("  Open your browser and go to: http://127.0.0.1:5000")
#     print("  Press CTRL+C to stop the server")
#     print("=" * 60)

#     # debug=True → auto-reloads when you change code (useful for development)
#     app.run(debug=True, host='0.0.0.0', port=5000)


import os
import pickle
import sqlite3
from flask import Flask, render_template, request, jsonify
from feature_extractor import extract_features, FEATURE_NAMES
from database import init_db, save_scan

app = Flask(__name__)

MODEL_PATH = "model/phishing_model.pkl"
model_data = None

init_db()

def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
    else:
        print("Model not found!")

load_model()

# ───────── HOME ─────────
@app.route('/')
def home():
    acc = None
    if model_data:
        acc = round(model_data['accuracy'] * 100, 2)
    return render_template('index.html', model_accuracy=acc)

# ───────── PREDICT ─────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'Enter URL'}), 400

    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    features = extract_features(url)
    model = model_data['model']

    pred = model.predict([features])[0]
    probs = model.predict_proba([features])[0]

    safe_prob = round(float(probs[0]) * 100, 2)
    phish_prob = round(float(probs[1]) * 100, 2)

    is_phishing = bool(pred == 1)

    # SAVE HISTORY
    save_scan(url, 'PHISHING' if is_phishing else 'SAFE', phish_prob)

    return jsonify({
        'url': url,
        'is_phishing': is_phishing,
        'safe_probability': safe_prob,
        'phishing_probability': phish_prob
    })

# ───────── HISTORY ─────────
@app.route('/history')
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    query = request.args.get('q')

    if query:
        c.execute("SELECT * FROM scans WHERE url LIKE ?", ('%' + query + '%',))
    else:
        c.execute("SELECT * FROM scans ORDER BY id DESC")

    data = c.fetchall()
    conn.close()

    return render_template('history.html', data=data)

# ───────── DASHBOARD ─────────
@app.route('/dashboard')
def dashboard():
    import sqlite3
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM scans")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE prediction='PHISHING'")
    phishing = c.fetchone()[0]

    safe = total - phishing

    conn.close()

    return render_template('dashboard.html',
                           total=total,
                           phishing=phishing,
                           safe=safe)

# ───────── OTHER PAGES ─────────
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/usecase')
def usecase():
    return render_template('usecase.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/download_guide')
def download_guide():
    return "Guide coming soon!"

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter

@app.route('/download_report')
def download_report():
    import sqlite3

    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT url, prediction, phishing_prob, date FROM scans ORDER BY id DESC")
    data = c.fetchall()
    conn.close()

    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)

    elements = []

    elements.append(Paragraph("Phishing Detection Report"))
    elements.append(Spacer(1, 10))

    for row in data[:20]:
        text = f"{row[0]} - {row[1]} ({row[2]}%) - {row[3]}"
        elements.append(Paragraph(text))
        elements.append(Spacer(1, 5))

    doc.build(elements)

    return "✅ Report generated! Check your project folder."

# ───────── RUN ─────────
if __name__ == '__main__':
    app.run(debug=True)
