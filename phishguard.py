"""
PhishGuard — Explainable Phishing Detection Agent
Educational prototype for BSIT 3103. Standard library only (Python 3.8+).

What is real: Naive Bayes training/inference, explanations, similarity, rules.
What is simulated: the dataset is fictional; the feedback loop retrains in-session only.
"""

import json
import math
import os
import re
from collections import Counter

DISCLAIMER = """
================================================================
 PhishGuard — Educational Phishing Detection Prototype
 Student project for learning purposes only.
 It does NOT claim a message is safe or malicious with certainty.
 Every result is a FLAG FOR HUMAN REVIEW, not a verdict.
================================================================
"""

TRAINING_DATA = [
    # ---- PHISHING (fictional) ----
    {"text": "Urgent: Your account will be suspended. Click http://secure-verify-login.com to confirm your password now.", "label": "phishing"},
    {"text": "Congratulations! You won a $500 gift card. Claim at http://bit.ly/free-prize-now", "label": "phishing"},
    {"text": "Dear customer, we detected unusual activity. Verify your identity at http://bank-secure-alert.net immediately.", "label": "phishing"},
    {"text": "Final notice: unpaid invoice. Pay within 24 hours at http://pay-now-invoices.com or legal action follows.", "label": "phishing"},
    {"text": "Your package is held. Pay the $2.99 delivery fee at http://track-parcel-fee.info", "label": "phishing"},
    {"text": "IT support: Your mailbox is full. Re-enter your credentials at http://webmail-reset.support to avoid deletion.", "label": "phishing"},
    {"text": "You have a pending refund of $840. Submit bank details at http://tax-refund-claim.org", "label": "phishing"},
    {"text": "Verify your account now or lose access permanently. http://account-verify-secure.com/login", "label": "phishing"},
    {"text": "Exclusive offer for you only! Act now, limited time. http://cheap-meds-online.biz", "label": "phishing"},
    {"text": "Your Netflix payment failed. Update card details at http://netflix-billing-update.co", "label": "phishing"},
    {"text": "HR notice: Salary revision attached. Open http://hr-docs-secure.net to view.", "label": "phishing"},
    {"text": "Your Apple ID was used to sign in from an unknown device. Confirm at http://appleid-verify-alert.com", "label": "phishing"},
    {"text": "You are selected for a work-from-home job. Send your ID and bank info to hiring@quickjobs-now.xyz", "label": "phishing"},
    {"text": "Warning: virus detected on your device. Call support and visit http://pc-clean-now.help", "label": "phishing"},
    {"text": "Reset your password immediately using this link: http://password-reset-secure.top", "label": "phishing"},
    {"text": "Claim your inheritance of $2,000,000. Contact barrister at http://inheritance-claim.biz", "label": "phishing"},
    {"text": "Your subscription auto-renewed for $99.99. Cancel here: http://cancel-subscription-now.info", "label": "phishing"},
    {"text": "Security alert: someone changed your PIN. Verify at http://bank-pin-verify.net", "label": "phishing"},
    {"text": "Dear user, your email storage is 99% full. Click http://email-storage-fix.com to upgrade free.", "label": "phishing"},
    {"text": "Limited slots: free cryptocurrency giveaway. Send your wallet seed phrase to claim at http://crypto-airdrop-claim.io", "label": "phishing"},

    # ---- LEGITIMATE (fictional) ----
    {"text": "Hi team, the meeting is moved to 3 PM tomorrow in room 204. Thanks, Sarah.", "label": "legit"},
    {"text": "Your order #4821 has shipped and will arrive Tuesday. Track it in your account dashboard.", "label": "legit"},
    {"text": "Reminder: library books are due on Friday. Renew online through the student portal.", "label": "legit"},
    {"text": "Hello, I reviewed the report and added comments. Let me know if you have questions.", "label": "legit"},
    {"text": "The cafeteria will be closed on Monday for maintenance. Sorry for the inconvenience.", "label": "legit"},
    {"text": "Your appointment with Dr. Lee is confirmed for Nov 20 at 10 AM. Reply to reschedule.", "label": "legit"},
    {"text": "Thanks for your submission. We will review it and respond within five business days.", "label": "legit"},
    {"text": "Class is cancelled today due to weather. Check the portal for updates.", "label": "legit"},
    {"text": "Your monthly statement is available in your online banking account. No action needed.", "label": "legit"},
    {"text": "Please find attached the agenda for next week's department meeting.", "label": "legit"},
    {"text": "Congratulations on finishing the semester! Grades will be posted by Friday.", "label": "legit"},
    {"text": "The IT maintenance window is scheduled for Saturday 2-4 AM. Services may be briefly unavailable.", "label": "legit"},
    {"text": "Your ride is arriving in 3 minutes. Driver: Alex, silver sedan.", "label": "legit"},
    {"text": "We received your support request. Ticket #5521 has been created.", "label": "legit"},
    {"text": "Happy birthday! Have a great day from all of us on the team.", "label": "legit"},
    {"text": "The gym will have reduced hours during the holiday break.", "label": "legit"},
    {"text": "Your password was changed successfully. If this was not you, contact the IT helpdesk.", "label": "legit"},
    {"text": "Lunch is provided at the workshop. Please confirm dietary restrictions by Wednesday.", "label": "legit"},
    {"text": "The bus route 12 will be detoured next week due to road work.", "label": "legit"},
    {"text": "Thank you for attending the seminar. The slides are posted on the course page.", "label": "legit"},
]

STOPWORDS = set("a an the and or but if then to of in on at for with is are was were be been being "
                "this that these those your you i we they it he she my our their".split())

URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
URGENCY_WORDS = {"urgent", "immediately", "now", "final", "warning", "alert", "suspend",
                 "suspended", "verify", "confirm", "expire", "expired", "limited", "act",
                 "click", "locked", "delete", "terminated"}
CREDENTIAL_WORDS = {"password", "pin", "credentials", "login", "seed", "phrase", "ssn",
                    "bank", "card", "identity", "otp"}
SUSPICIOUS_URL_PARTS = ["bit.ly", "tinyurl", "t.co", ".xyz", ".top", ".biz", ".info", ".click", ".help"]


def extract_urls(text):
    return URL_PATTERN.findall(text.lower())


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS]


class NaiveBayesPhish:
    """Multinomial Naive Bayes with Laplace smoothing, implemented from scratch."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_word_counts = {}
        self.class_totals = {}
        self.class_docs = {}
        self.vocab = set()
        self.n_docs = 0

    def train(self, data):
        self.class_word_counts = {"phishing": Counter(), "legit": Counter()}
        self.class_totals = {"phishing": 0, "legit": 0}
        self.class_docs = {"phishing": 0, "legit": 0}
        self.vocab = set()
        self.n_docs = 0
        for item in data:
            label = item["label"]
            self.class_docs[label] += 1
            self.n_docs += 1
            for t in tokenize(item["text"]):
                self.class_word_counts[label][t] += 1
                self.class_totals[label] += 1
                self.vocab.add(t)

    def _log_prior(self, label):
        return math.log(self.class_docs[label] / self.n_docs)

    def _log_likelihood(self, token, label):
        count = self.class_word_counts[label].get(token, 0)
        total = self.class_totals[label]
        V = len(self.vocab)
        return math.log((count + self.alpha) / (total + self.alpha * V))

    def score(self, text):
        tokens = tokenize(text)
        log_scores = {}
        for label in ("phishing", "legit"):
            s = self._log_prior(label)
            for t in tokens:
                s += self._log_likelihood(t, label)
            log_scores[label] = s
        # log-sum-exp to avoid underflow
        max_s = max(log_scores.values())
        exp_ph = math.exp(log_scores["phishing"] - max_s)
        exp_le = math.exp(log_scores["legit"] - max_s)
        prob_phishing = exp_ph / (exp_ph + exp_le)
        return prob_phishing, log_scores

    def explain(self, text, top_k=6):
        """Return tokens ranked by how strongly they pushed the decision."""
        tokens = tokenize(text)
        contributions = {}
        for t in set(tokens):
            contributions[t] = self._log_likelihood(t, "phishing") - self._log_likelihood(t, "legit")
        ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:top_k]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def find_similar(text, data, k=2):
    tokens = tokenize(text)
    scored = [(jaccard(tokens, tokenize(d["text"])), d) for d in data]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for s in scored[:k] if s[0] > 0]


def rule_signals(text):
    low = text.lower()
    urls = extract_urls(text)
    tokens = set(tokenize(text))
    signals = []
    if urls:
        signals.append(f"{len(urls)} URL(s) present in the message")
        for u in urls:
            if any(p in u for p in SUSPICIOUS_URL_PARTS):
                signals.append(f"URL '{u}' uses a link shortener or uncommon domain ending")
    urgency = tokens & URGENCY_WORDS
    if len(urgency) >= 2:
        signals.append(f"Urgency language: {', '.join(sorted(urgency))}")
    creds = tokens & CREDENTIAL_WORDS
    if creds:
        signals.append(f"Credential/payment language: {', '.join(sorted(creds))}")
    return signals


def decide(text, data, model, threshold=0.65):
    prob_phishing, _ = model.score(text)
    label = "phishing" if prob_phishing >= 0.5 else "legit"
    confidence = max(prob_phishing, 1 - prob_phishing)
    signals = rule_signals(text)
    urls = extract_urls(text)

    review_reasons = []
    if confidence < threshold:
        review_reasons.append(f"Low model confidence ({confidence:.2f} below the {threshold} threshold)")
    if label == "legit" and urls and any(w in text.lower() for w in
                                         ["verify", "confirm", "login", "password", "bank", "pay"]):
        review_reasons.append("Model says 'legit' but the message combines a URL with verification/payment language")
    if len(signals) >= 3:
        review_reasons.append("Three or more independent rule-based risk signals fired")

    return {
        "label": label,
        "prob_phishing": prob_phishing,
        "confidence": confidence,
        "signals": signals,
        "explanation": model.explain(text),
        "similar": find_similar(text, data),
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


FEEDBACK_FILE = "feedback.json"


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return []


def save_feedback(entry):
    fb = load_feedback()
    fb.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(fb, f, indent=2)


def print_result(result):
    print("\n--- ANALYSIS ---")
    print(f"Model verdict : {result['label'].upper()}")
    print(f"P(phishing)   : {result['prob_phishing']:.3f}   |   confidence: {result['confidence']:.2f}")

    print("\nWhy this verdict (top contributing tokens):")
    for token, contribution in result["explanation"]:
        direction = "pushes toward PHISHING" if contribution > 0 else "pushes toward LEGIT"
        print(f"   {token:<18} {contribution:+.3f}  {direction}")

    if result["signals"]:
        print("\nRule-based signals:")
        for s in result["signals"]:
            print(f"   - {s}")

    if result["similar"]:
        print("\nMost similar messages in the training set:")
        for score, item in result["similar"]:
            print(f"   [{item['label']:<8}] similarity={score:.2f}  {item['text'][:65]}...")

    if result["needs_review"]:
        print("\n[!] FLAGGED FOR HUMAN REVIEW:")
        for r in result["review_reasons"]:
            print(f"   - {r}")
    else:
        print("\nNo human-review triggers were raised by the prototype rules.")

    print("\nReminder: this is a flag for review, not a certainty.")


def main():
    print(DISCLAIMER)
    data = TRAINING_DATA + load_feedback()
    model = NaiveBayesPhish()
    model.train(data)
    print(f"Trained on {len(data)} messages ({len(model.vocab)} unique tokens).")
    print("Type a message to analyze, or 'quit' to exit.\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break

        result = decide(text, data, model)
        print_result(result)

        feedback = input("\nWas this result correct? [y]es / [n]o / [s]kip: ").strip().lower()
        if feedback == "n":
            true_label = input("Correct label? [phishing/legit]: ").strip().lower()
            if true_label in {"phishing", "legit"}:
                save_feedback({"text": text, "label": true_label})
                print("Feedback saved. Retraining now (simulated feedback loop)...")
                data = TRAINING_DATA + load_feedback()
                model = NaiveBayesPhish()
                model.train(data)
                print(f"Retrained on {len(data)} messages.")


if __name__ == "__main__":
    main()
