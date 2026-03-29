import os
import pickle
import sqlite3
from flask import Flask, render_template, request, jsonify
from feature_extractor import extract_features, FEATURE_NAMES
from database import init_db, save_scan

app = Flask(__name__)

MODEL_PATH = "model/phishing_model.pkl"
model_data = None

# INIT DB
init_db()

# LOAD MODEL
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
    try:
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

        # SAVE
        save_scan(url, 'PHISHING' if is_phishing else 'SAFE', phish_prob)

        return jsonify({
            'url': url,
            'prediction': 'PHISHING' if is_phishing else 'SAFE',
            'safe_probability': safe_prob,
            'phishing_probability': phish_prob,
            'risk_level': (
                "HIGH" if phish_prob >= 80 else
                "MEDIUM" if phish_prob >= 50 else
                "LOW"
            )
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ───────── DASHBOARD ─────────
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    # TOTAL
    c.execute("SELECT COUNT(*) FROM scans")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE prediction='PHISHING'")
    phishing = c.fetchone()[0]

    safe = total - phishing

    # RECENT
    c.execute("""
        SELECT url, prediction, phishing_prob, date
        FROM scans
        ORDER BY id DESC
        LIMIT 5
    """)
    recent = c.fetchall()

    # RISK %
    risk_percent = round((phishing / total) * 100, 2) if total > 0 else 0

    # TREND (7 days)
    c.execute("""
        SELECT date(date), COUNT(*)
        FROM scans
        GROUP BY date(date)
        ORDER BY date(date) DESC
        LIMIT 7
    """)
    trend = c.fetchall()
    trend.reverse()

    dates = [row[0] for row in trend]
    counts = [row[1] for row in trend]

    conn.close()

    return render_template(
        'dashboard.html',
        total=total,
        phishing=phishing,
        safe=safe,
        recent=recent,
        risk_percent=risk_percent,
        dates=dates,
        counts=counts
    )


# ───────── HISTORY ─────────
@app.route('/history')
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("SELECT * FROM scans ORDER BY id DESC")
    data = c.fetchall()

    conn.close()

    return render_template('history.html', data=data)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '').lower()

    if "phishing" in msg:
        reply = "Phishing is a fake website that tries to steal your data."
    elif "safe" in msg:
        reply = "Safe websites are trusted and secure."
    elif "how to use" in msg:
        reply = "Go to home page, enter URL and click scan."
    elif "model" in msg:
        reply = "This tool uses Machine Learning (Random Forest)."
    else:
        reply = "I am your AI assistant 🤖. Ask anything about this project."

    return jsonify({'reply': reply})
# ───────── HELP ─────────
@app.route('/help')
def help_page():
    return render_template('help.html')


# ───────── RUN ─────────
if __name__ == '__main__':
    app.run(debug=True)