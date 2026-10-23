"""
Train the TF-IDF + Logistic Regression scam classifier and export it for the web app.

USAGE (Google Colab or a PC with Python 3 + scikit-learn):
    python train_model.py --data SMSSpamCollection phishing.csv --out .

--data accepts one or more files:
  * UCI "SMSSpamCollection" (tab separated: label<TAB>text, labels ham/spam)
  * any CSV with a text column and a label column (see --text-col / --label-col)
Writes: model.js (loaded by the web app) and metrics.json (use these REAL numbers in your documentation).
"""
import argparse, csv, json, re, sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

SEED = 42
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:ly|gl|co|xyz|top|click|link|info|site|online|icu|tk|cc|buzz|work|club|ph|com|net|org)(?:/\S*)?)")
MONEY_RE = re.compile(r"(?:\u20b1|\$|php\s?)\d[\d,]*(?:\.\d+)?")
NUM_RE = re.compile(r"\+?\d[\d\s\-]{6,}\d")

def preprocess(text):
    """MUST match preprocess() in engine.js exactly."""
    t = str(text).lower()
    t = URL_RE.sub(" urltoken ", t)
    t = MONEY_RE.sub(" moneytoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

SAFE = {"ham", "safe", "legitimate", "legit", "benign", "not spam", "0", "potentially safe"}
BAD = {"spam", "phishing", "scam", "suspicious", "smishing", "phishing email", "1", "fraud"}

def load(path, text_col=None, label_col=None):
    rows, skipped = [], 0
    csv.field_size_limit(10**8)
    if "smsspamcollection" in os.path.basename(path).lower() and not path.lower().endswith(".csv"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "\t" not in line: continue
                lab, txt = line.rstrip("\n").split("\t", 1)
                rows.append((txt, 1 if lab.strip().lower() == "spam" else 0))
        return rows, skipped
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in rd.fieldnames}
        tc = text_col or next((cols[k] for k in ("text", "message", "body", "email text", "content", "sms") if k in cols), None)
        lc = label_col or next((cols[k] for k in ("label", "class", "type", "category", "email type", "v1") if k in cols), None)
        if not tc or not lc:
            sys.exit(f"{path}: cannot find text/label columns in {rd.fieldnames}. Use --text-col / --label-col.")
        for r in rd:
            lab = str(r[lc]).strip().lower(); txt = r[tc]
            if not txt or not txt.strip(): skipped += 1; continue
            if lab in SAFE: rows.append((txt, 0))
            elif lab in BAD: rows.append((txt, 1))
            else: skipped += 1
    return rows, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--text-col"); ap.add_argument("--label-col")
    a = ap.parse_args()

    data = []
    for p in a.data:
        r, s = load(p, a.text_col, a.label_col)
        print(f"{p}: {len(r)} rows loaded, {s} skipped"); data += r
    seen, X, y = set(), [], []
    for t, l in data:                       # remove duplicates (after preprocessing)
        k = preprocess(t)
        if not k or k in seen: continue
        seen.add(k); X.append(t); y.append(l)
    y = np.array(y)
    print(f"After removing duplicates: {len(X)} messages | Safe={int((y==0).sum())} Suspicious={int((y==1).sum())}")

    idx = np.arange(len(X))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    pick = lambda ii: ([X[i] for i in ii], y[ii])
    Xtr, ytr = pick(i_tr); Xva, yva = pick(i_va); Xte, yte = pick(i_te)
    print(f"Split -> train {len(Xtr)} | validation {len(Xva)} | test {len(Xte)}")
    if len(Xte) < 200:
        print("WARNING: test set is tiny; these metrics are NOT reliable. Use the real dataset for your documentation.")

    best = None
    for ng in [(1, 1), (1, 2)]:
        for C in [0.1, 0.3, 1, 3, 10, 30]:
            vec = TfidfVectorizer(preprocessor=preprocess, ngram_range=ng, sublinear_tf=True, min_df=1 if len(Xtr) < 1000 else 2, max_features=10000)
            A = vec.fit_transform(Xtr)
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED).fit(A, ytr)
            f1 = f1_score(yva, clf.predict(vec.transform(Xva)), pos_label=1, zero_division=0)
            if best is None or f1 >= best[0]:  # ties go to the larger C
                best = (f1, ng, C, vec, clf)
    f1v, ng, C, vec, clf = best
    print(f"Chosen on validation: ngram={ng}, C={C}, validation F1(Suspicious)={f1v:.3f}")

    pred = clf.predict(vec.transform(Xte))
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    rep = classification_report(yte, pred, target_names=["Potentially Safe", "Suspicious"], output_dict=True, zero_division=0)
    print("\nConfusion matrix (rows=true, cols=pred) [Safe, Suspicious]:\n", cm)
    print(classification_report(yte, pred, target_names=["Potentially Safe", "Suspicious"], digits=3, zero_division=0))
    print("Accuracy:", round(accuracy_score(yte, pred), 4))

    terms = {}
    idf, coef = vec.idf_, clf.coef_[0]
    for term, i in vec.vocabulary_.items():
        terms[term] = [round(float(idf[i]), 4), round(float(coef[i]), 4)]
    model = {"version": 1, "ngram": list(ng), "intercept": round(float(clf.intercept_[0]), 4),
             "terms": terms, "info": {"C": C, "n_train": len(Xtr), "n_terms": len(terms)}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "model.js"), "w", encoding="utf-8") as f:
        f.write("var SCAM_MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n"
                "if (typeof module !== 'undefined') module.exports = SCAM_MODEL;\n")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"counts": {"total": len(X), "train": len(Xtr), "validation": len(Xva), "test": len(Xte),
                              "safe": int((y == 0).sum()), "suspicious": int((y == 1).sum())},
                   "config": {"ngram": list(ng), "C": C, "seed": SEED},
                   "confusion_matrix_[safe,suspicious]": cm.tolist(), "report": rep,
                   "accuracy": float(accuracy_score(yte, pred))}, f, indent=2)
    print(f"\nWrote {a.out}/model.js ({len(terms)} terms) and metrics.json")

if __name__ == "__main__":
    main()
