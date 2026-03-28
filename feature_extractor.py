"""
feature_extractor.py
====================
This file extracts FEATURES from a URL.

What is a "Feature"?
--------------------
Think of it like a doctor checking symptoms. Instead of checking
"fever, cough, cold", we check URL properties like:
  - How long is the URL?
  - Does it have an IP address instead of a domain name?
  - How many dots, dashes are in it?
  - Does it use HTTPS?
  - etc.

The ML model learns: "phishing URLs usually have THESE patterns"
"""

import re
import urllib.parse


def extract_features(url: str) -> list:
    """
    Given a URL string, return a list of numbers (features).
    The ML model uses these numbers to decide: phishing or safe?
    """

    features = []

    # ── 1. URL LENGTH ──────────────────────────────────────────────────────────
    # Phishing URLs tend to be very long (to hide suspicious parts)
    features.append(len(url))

    # ── 2. COUNT OF DOTS ───────────────────────────────────────────────────────
    # e.g. "secure.login.bank.com.evil.com" has many dots → suspicious
    features.append(url.count('.'))

    # ── 3. COUNT OF HYPHENS ────────────────────────────────────────────────────
    # e.g. "paypal-login-secure-verify.com" → lots of hyphens = red flag
    features.append(url.count('-'))

    # ── 4. COUNT OF UNDERSCORES ────────────────────────────────────────────────
    features.append(url.count('_'))

    # ── 5. COUNT OF SLASHES ────────────────────────────────────────────────────
    features.append(url.count('/'))

    # ── 6. COUNT OF QUESTION MARKS ─────────────────────────────────────────────
    # Query strings like "?redirect=evil.com"
    features.append(url.count('?'))

    # ── 7. COUNT OF EQUALS SIGNS ───────────────────────────────────────────────
    features.append(url.count('='))

    # ── 8. COUNT OF @ SYMBOL ──────────────────────────────────────────────────
    # "http://user@evil.com" — the browser ignores everything before @
    # This is a classic phishing trick!
    features.append(url.count('@'))

    # ── 9. COUNT OF AMPERSANDS ─────────────────────────────────────────────────
    features.append(url.count('&'))

    # ── 10. COUNT OF EXCLAMATION MARKS ─────────────────────────────────────────
    features.append(url.count('!'))

    # ── 11. COUNT OF SPACES ────────────────────────────────────────────────────
    features.append(url.count(' '))

    # ── 12. COUNT OF TILDE ─────────────────────────────────────────────────────
    features.append(url.count('~'))

    # ── 13. COUNT OF COMMA ─────────────────────────────────────────────────────
    features.append(url.count(','))

    # ── 14. COUNT OF PLUS ──────────────────────────────────────────────────────
    features.append(url.count('+'))

    # ── 15. COUNT OF ASTERISK ──────────────────────────────────────────────────
    features.append(url.count('*'))

    # ── 16. COUNT OF HASH / FRAGMENT ──────────────────────────────────────────
    features.append(url.count('#'))

    # ── 17. COUNT OF DOLLAR ────────────────────────────────────────────────────
    features.append(url.count('$'))

    # ── 18. COUNT OF PERCENT ───────────────────────────────────────────────────
    # Percent-encoded URLs to hide malicious content
    features.append(url.count('%'))

    # ── 19. HAS IP ADDRESS? ────────────────────────────────────────────────────
    # "http://192.168.1.1/login" — real banks don't use raw IP addresses
    has_ip = 1 if re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])', url
    ) else 0
    features.append(has_ip)

    # ── 20. USES HTTPS? ────────────────────────────────────────────────────────
    # HTTPS = encrypted connection (safer, but phishing sites also use it now)
    has_https = 1 if url.lower().startswith('https') else 0
    features.append(has_https)

    # ── 21. DOMAIN LENGTH ──────────────────────────────────────────────────────
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        features.append(len(domain))
    except Exception:
        features.append(0)

    # ── 22. PATH LENGTH ────────────────────────────────────────────────────────
    try:
        features.append(len(parsed.path))
    except Exception:
        features.append(0)

    # ── 23. NUMBER OF SUBDOMAINS ───────────────────────────────────────────────
    # "login.secure.paypal.com.evil.com" → many subdomains
    try:
        subdomain_count = len(domain.split('.')) - 2
        features.append(max(0, subdomain_count))
    except Exception:
        features.append(0)

    # ── 24. HAS "LOGIN" IN URL? ─────────────────────────────────────────────────
    # Phishers often add words like login, secure, verify to look real
    features.append(1 if 'login' in url.lower() else 0)

    # ── 25. HAS "SECURE" IN URL? ───────────────────────────────────────────────
    features.append(1 if 'secure' in url.lower() else 0)

    # ── 26. HAS "ACCOUNT" IN URL? ──────────────────────────────────────────────
    features.append(1 if 'account' in url.lower() else 0)

    # ── 27. HAS "VERIFY" IN URL? ───────────────────────────────────────────────
    features.append(1 if 'verify' in url.lower() else 0)

    # ── 28. HAS "UPDATE" IN URL? ───────────────────────────────────────────────
    features.append(1 if 'update' in url.lower() else 0)

    # ── 29. HAS "BANKING" IN URL? ──────────────────────────────────────────────
    features.append(1 if 'banking' in url.lower() else 0)

    # ── 30. HAS "CONFIRM" IN URL? ──────────────────────────────────────────────
    features.append(1 if 'confirm' in url.lower() else 0)

    # ── 31. URL ENTROPY (Randomness) ───────────────────────────────────────────
    # Random-looking URLs are often generated by phishing tools
    import math
    def entropy(s):
        if not s:
            return 0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        return -sum((f/len(s)) * math.log2(f/len(s)) for f in freq.values())
    features.append(round(entropy(url), 4))

    # ── 32. COUNT OF DIGITS IN URL ─────────────────────────────────────────────
    features.append(sum(c.isdigit() for c in url))

    return features


# Column names — must match the order above
FEATURE_NAMES = [
    "url_length", "count_dots", "count_hyphens", "count_underscores",
    "count_slashes", "count_question_marks", "count_equals", "count_at",
    "count_ampersand", "count_exclamation", "count_spaces", "count_tilde",
    "count_comma", "count_plus", "count_asterisk", "count_hash",
    "count_dollar", "count_percent", "has_ip_address", "has_https",
    "domain_length", "path_length", "num_subdomains",
    "has_login", "has_secure", "has_account", "has_verify",
    "has_update", "has_banking", "has_confirm",
    "url_entropy", "count_digits"
]
