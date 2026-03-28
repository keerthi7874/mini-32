"""
train_model.py
==============
This script TRAINS the Machine Learning model.

Simple Explanation:
-------------------
Imagine teaching a child to spot fake currency notes.
You show them 1000 real notes and 1000 fake notes.
After seeing enough examples, the child learns the PATTERNS.

Similarly, we show our ML model:
  - Many PHISHING URLs  (label = 1 = bad)
  - Many SAFE URLs      (label = 0 = good)

The model learns patterns → gets saved → Flask app uses it to predict!

How to Run:
-----------
    python train_model.py

This will:
1. Download/use phishing URL dataset
2. Extract features from each URL
3. Train a Random Forest model
4. Save it as model/phishing_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from feature_extractor import extract_features, FEATURE_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load phishing URL dataset.

    Your Kaggle CSV has:
      - Column name : URL   (capital letters)
      - Label values: bad   (phishing) and good (safe)

    This function handles that exact format.
    """

    csv_path = "dataset.csv"

    if os.path.exists(csv_path):
        print(f"✅ Found {csv_path} — loading it...")
        df = pd.read_csv(csv_path)

        print(f"   Columns found: {list(df.columns)}")
        print(f"   Total rows: {len(df)}")

        # ── Step 1: Find URL column (handles URL, url, Url, website, link)
        url_col = None
        for col in df.columns:
            if col.strip().lower() in ['url', 'urls', 'website', 'link']:
                url_col = col
                break

        # ── Step 2: Find Label column (handles Label, label, status, type)
        label_col = None
        for col in df.columns:
            if col.strip().lower() in ['label', 'labels', 'status', 'type', 'class', 'phishing']:
                label_col = col
                break

        if url_col and label_col:
            print(f"   URL column   : '{url_col}'")
            print(f"   Label column : '{label_col}'")

            df = df[[url_col, label_col]].rename(
                columns={url_col: 'url', label_col: 'label'}
            )

            # Show what unique label values exist
            unique_labels = df['label'].unique()
            print(f"   Label values found: {unique_labels}")

            # ── Step 3: Convert ANY label format to 0 and 1
            # Handles: bad/good, phishing/legitimate, 1/0, yes/no, malicious/safe
            def convert_label(x):
                x = str(x).strip().lower()
                if x in ['bad', 'phishing', '1', 'malicious', 'yes', 'phish', 'unsafe']:
                    return 1   # phishing
                else:
                    return 0   # safe (good, legitimate, 0, no, safe)

            df['label'] = df['label'].map(convert_label)
            df = df.dropna()

            phishing_count = df['label'].sum()
            safe_count = (df['label'] == 0).sum()
            print(f"   ✅ Phishing URLs : {phishing_count}")
            print(f"   ✅ Safe URLs     : {safe_count}")

            # Balance the dataset if very unequal
            # (too many of one type confuses the model)
            if phishing_count > 0 and safe_count > 0:
                min_count = min(phishing_count, safe_count)
                if phishing_count / safe_count > 3 or safe_count / phishing_count > 3:
                    print(f"   ⚖️  Balancing dataset to {min_count} each...")
                    df_phish = df[df['label'] == 1].sample(min_count, random_state=42)
                    df_safe  = df[df['label'] == 0].sample(min_count, random_state=42)
                    df = pd.concat([df_phish, df_safe]).sample(frac=1, random_state=42)
                    print(f"   Balanced to {len(df)} total URLs")

            return df
        else:
            print(f"⚠️  Could not find URL or Label columns in: {list(df.columns)}")
            print("   Using synthetic data instead...")

    # ── SYNTHETIC DATA (if no CSV) ─────────────────────────────────────────────
    print("📦 Generating synthetic training data (1000 phishing + 1000 safe URLs)...")

    # Real-looking SAFE URLs
    safe_urls = [
        "https://www.google.com",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://github.com/user/repo",
        "https://stackoverflow.com/questions/12345",
        "https://www.wikipedia.org/wiki/Python",
        "https://docs.python.org/3/tutorial/",
        "https://www.amazon.in/product/laptop",
        "https://mail.google.com/mail/u/0/",
        "https://www.flipkart.com/laptops",
        "https://www.linkedin.com/in/username",
        "https://www.twitter.com/home",
        "https://www.reddit.com/r/learnpython",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.npmjs.com/package/react",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    ]

    # Suspicious-looking PHISHING URLs
    phishing_urls = [
        "http://192.168.1.1/paypal/login/secure/account/verify",
        "http://www.paypa1.com-secure-login.xyz/account/verify",
        "http://secure.banking.login-verify.tk/update?user=admin",
        "http://192.168.0.100/bank/login?redirect=evil.com@phish.net",
        "http://login-secure-paypal-verify.000webhostapp.com/",
        "http://update-your-account.secure-login.ml/banking/verify",
        "http://www.g00gle.com-secure.tk/login?next=/account",
        "http://sbi-online-secure.login-verify.cf/banking/update",
        "http://accounts-login-secure.update-verify.gq/confirm",
        "http://facebook-login.secure-verify-account.tk/confirm",
        "http://hdfc-netbanking-secure.ml/login?userid=&password=",
        "http://paypal-account-verify-secure.000webhostapp.com/update",
        "http://amazon-india-order-confirm.xyz/login?token=1234567",
        "http://icici-bank-login.secure-update.cf/netbanking/verify",
        "http://ebay-login-confirm-account-secure.tk/update?ref=abc",
    ]

    # Generate more varied examples
    import random
    random.seed(42)

    domains = ['google', 'youtube', 'github', 'wikipedia', 'amazon',
               'stackoverflow', 'linkedin', 'twitter', 'flipkart', 'reddit',
               'coursera', 'udemy', 'mozilla', 'apple', 'microsoft']
    tlds = ['.com', '.org', '.net', '.in', '.edu', '.gov', '.io']
    paths = ['/home', '/about', '/products', '/services', '/contact',
             '/docs/guide', '/api/v1/users', '/blog/post-title', '/shop']

    all_safe = list(safe_urls)
    for _ in range(985):
        d = random.choice(domains)
        t = random.choice(tlds)
        p = random.choice(paths)
        scheme = random.choice(['https://', 'https://www.'])
        all_safe.append(f"{scheme}{d}{t}{p}")

    bad_words = ['login', 'secure', 'verify', 'account', 'update',
                 'banking', 'confirm', 'signin', 'password']
    bad_tlds = ['.tk', '.ml', '.cf', '.gq', '.xyz', '.ga']
    bad_hosts = ['000webhostapp.com', 'weebly.com', 'wix.com', 'blogspot.com']

    all_phish = list(phishing_urls)
    for _ in range(985):
        bw1 = random.choice(bad_words)
        bw2 = random.choice(bad_words)
        bt = random.choice(bad_tlds)
        scheme = random.choice(['http://', 'http://www.'])
        style = random.randint(0, 3)
        if style == 0:
            url = f"{scheme}{bw1}-{bw2}{bt}/{bw1}?id={random.randint(100,999)}"
        elif style == 1:
            ip = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
            url = f"http://{ip}/{bw1}/{bw2}/confirm"
        elif style == 2:
            host = random.choice(bad_hosts)
            url = f"{scheme}{bw1}-{bw2}-secure.{host}/{bw1}"
        else:
            url = f"{scheme}secure-{bw1}.{bw2}-verify{bt}/"
        all_phish.append(url)

    urls = all_safe[:1000] + all_phish[:1000]
    labels = [0] * 1000 + [1] * 1000

    df = pd.DataFrame({'url': urls, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Generated {len(df)} URLs (1000 safe + 1000 phishing)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: EXTRACT FEATURES FROM EACH URL
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df):
    """
    For each URL in the dataframe, extract numerical features.
    
    Input:  DataFrame with 'url' column
    Output: NumPy array of shape (n_urls, n_features)
    """
    print("\n⚙️  Extracting features from URLs...")
    X = []
    for i, url in enumerate(df['url']):
        try:
            features = extract_features(str(url))
            X.append(features)
        except Exception as e:
            # If a URL is weird/broken, use zeros
            X.append([0] * len(FEATURE_NAMES))

        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{len(df)} URLs...")

    return np.array(X)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN THE MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train(X, y):
    """
    Train a Random Forest Classifier.
    
    What is Random Forest?
    ----------------------
    Imagine asking 100 different experts to look at a URL and vote.
    Majority vote wins! That's basically Random Forest.
    
    Each "expert" = one Decision Tree
    100 experts = Random Forest (100 trees)
    
    Benefits:
    - Very accurate
    - Doesn't overfit easily
    - Works well with URL data
    """
    print("\n🌲 Training Random Forest model...")

    # Split data: 80% training, 20% testing
    # (We test on data the model has NEVER seen → honest accuracy!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} URLs")
    print(f"   Testing set:  {len(X_test)} URLs")

    # Create & train the model
    model = RandomForestClassifier(
        n_estimators=100,    # 100 decision trees
        max_depth=15,        # Each tree can be max 15 levels deep
        random_state=42,     # For reproducibility
        n_jobs=-1            # Use all CPU cores for speed
    )
    model.fit(X_train, y_train)
    print("   ✅ Model trained!")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Accuracy: {acc * 100:.2f}%")
    print(f"\n   Detailed Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=['Safe (0)', 'Phishing (1)']
    )
    print(report)

    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Safe  Phishing")
    print(f"   Actual Safe    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Actual Phish   {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Feature importance — which features matter most?
    print(f"\n🔍 Top 10 Most Important Features:")
    importances = model.feature_importances_
    feat_importance = sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    )
    for name, imp in feat_importance[:10]:
        bar = '█' * int(imp * 200)
        print(f"   {name:<25} {bar} ({imp:.4f})")

    return model, acc


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SAVE THE MODEL
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, accuracy):
    """Save the trained model to disk using pickle."""
    os.makedirs("model", exist_ok=True)
    model_path = "model/phishing_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({
            'model': model,
            'accuracy': accuracy,
            'feature_names': FEATURE_NAMES
        }, f)

    print(f"\n💾 Model saved to: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PHISHING WEBSITE DETECTOR — MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Extract features
    X = build_feature_matrix(df)
    y = df['label'].values

    # Step 3: Train model
    model, accuracy = train(X, y)

    # Step 4: Save model
    save_model(model, accuracy)

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE! Now run: python app.py")
    print("=" * 60)