import os
import pickle
import sqlite3
from flask import Flask, render_template, request, jsonify
from feature_extractor import extract_features, FEATURE_NAMES
from database import init_db, save_scan

app = Flask(__name__)

MODEL_PATH = "model/phishing_model.pkl"
model_data = None

# ───────── INIT DB ─────────
init_db()

# ───────── LOAD MODEL ─────────
def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print(f"✅ Model loaded! Accuracy: {model_data['accuracy']*100:.2f}%")
    else:
        print("❌ Model not found!")

load_model()

# ───────── HOME ─────────
@app.route('/')
def home():
    acc = None
    if model_data:
        acc = round(model_data['accuracy'] * 100, 2)
    return render_template('index.html', model_accuracy=acc)

# ───────── PREDICT (ORIGINAL WORKING LOGIC) ─────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'Enter URL'}), 400

        if not url.startswith(('http://', 'https://')):
            url_to_analyze = 'http://' + url
        else:
            url_to_analyze = url

        # Extract features
        features = extract_features(url_to_analyze)
        model = model_data['model']

        # ✅ ORIGINAL CORRECT LOGIC
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        safe_prob = round(float(probabilities[0]) * 100, 2)
        phish_prob = round(float(probabilities[1]) * 100, 2)

        is_phishing = bool(prediction == 1)

        # Feature details (for UI)
        feature_details = []
        for name, value in zip(FEATURE_NAMES, features):
            feature_details.append({
                'name': name.replace('_', ' ').title(),
                'value': round(float(value), 4),
                'suspicious': is_suspicious(name, value)
            })

        # Risk level
        if phish_prob >= 80:
            risk_level = "HIGH"
        elif phish_prob >= 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Save to DB
        save_scan(url, 'PHISHING' if is_phishing else 'SAFE', phish_prob)

        return jsonify({
            'url': url,
            'is_phishing': is_phishing,
            'prediction': 'PHISHING' if is_phishing else 'SAFE',
            'safe_probability': safe_prob,
            'phishing_probability': phish_prob,
            'risk_level': risk_level,
            'features': feature_details,
            'model_accuracy': round(model_data['accuracy'] * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ───────── SUSPICIOUS FEATURE LOGIC ─────────
def is_suspicious(feature_name, value):
    rules = {
        'url_length': value > 75,
        'count_dots': value > 4,
        'count_hyphens': value > 3,
        'count_at': value > 0,
        'has_ip_address': value == 1,
        'num_subdomains': value > 3,
        'count_percent': value > 3,
        'url_entropy': value > 4.5,
        'has_login': value == 1,
        'has_secure': value == 1,
        'has_verify': value == 1,
        'has_update': value == 1,
    }
    return rules.get(feature_name, False)


# ───────── DASHBOARD ─────────
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM scans")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE prediction='PHISHING'")
    phishing = c.fetchone()[0]

    safe = total - phishing

    c.execute("""
        SELECT url, prediction, phishing_prob, date
        FROM scans
        ORDER BY id DESC
        LIMIT 5
    """)
    recent = c.fetchall()

    risk_percent = round((phishing / total) * 100, 2) if total > 0 else 0

    conn.close()

    return render_template(
        'dashboard.html',
        total=total,
        phishing=phishing,
        safe=safe,
        recent=recent,
        risk_percent=risk_percent
    )


# ───────── HISTORY ─────────
@app.route('/history')
def history():
    search_query = request.args.get('q')

    conn = sqlite3.connect('history.db')   # ✅ correct DB
    cursor = conn.cursor()

    if search_query:
        cursor.execute("""
            SELECT * FROM scans
            WHERE url LIKE ?
            ORDER BY date DESC
        """, ('%' + search_query + '%',))
    else:
        cursor.execute("""
            SELECT * FROM scans
            ORDER BY date DESC
        """)

    data = cursor.fetchall()
    conn.close()

    return render_template('history.html', data=data)

# ───────── BATCH SCAN ─────────
@app.route('/batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    urls = data.get('urls', [])

    results = []
    model = model_data['model']

    for url in urls:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        try:
            features = extract_features(url)
            pred = model.predict([features])[0]
            probs = model.predict_proba([features])[0]

            results.append({
                'url': url,
                'is_phishing': bool(pred == 1),
                'phishing_probability': round(float(probs[1]) * 100, 2)
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})


# ───────── CHATBOT ─────────
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '').lower()

    if "phishing" in msg:
        reply = "Phishing is a fake website that tries to steal your data."
    elif "safe" in msg:
        reply = "Safe websites are trusted."
    elif "use" in msg:
        reply = "Enter a URL and click scan."
    else:
        reply = "I am your assistant 🤖"

    return jsonify({'reply': reply})


# ───────── HELP ─────────
@app.route('/help')
def help_page():
    return render_template('help.html')


# ───────── API INFO ─────────
@app.route('/api/info')
def api_info():
    return jsonify({
        'status': 'running',
        'model_loaded': model_data is not None,
        'accuracy': round(model_data['accuracy'] * 100, 2) if model_data else None
    })


# ───────── RUN ─────────
if __name__ == '__main__':
    app.run(debug=True)