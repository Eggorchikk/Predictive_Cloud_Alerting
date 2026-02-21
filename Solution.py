from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def main():
    np.random.seed(89)
    T = 2000

    metric, incidents = generate_data(T)

    X, y = make_window(metric, incidents, 20, 5)

    X_diff = np.diff(X, axis=1)
    X = np.concatenate([X, X_diff], axis=1)

    split_idx = int(0.75 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model_log = train_model(X_train, y_train, "log")
    model_rf = train_model(X_train, y_train, "rf")

    run_experiments(model_log, X_test, y_test, "log")
    run_experiments(model_rf, X_test, y_test, "rf")


def generate_data(T):
    metric = np.sin(np.linspace(0, 20, T)) + np.random.normal(0, 0.2, T)

    incidents = np.zeros(T)

    used_idx = set()
    drift_length = 5

    for i in range(40):
        while True:
            idx = np.random.randint(50, T-30)
            if all(j not in used_idx for j in range(idx, idx+5)):
                break
        metric[idx-drift_length:idx] += np.linspace(0, 1, drift_length)
        metric[idx:idx+5] += 5
        incidents[idx:idx+5] = 1
        used_idx.update(range(idx, idx+5))
    
    return metric, incidents


def make_window(data, incidents, W, H):
    X = []
    y = []

    for i in range (W, len(data)-H):
        X.append(data[i-W:i])
        y.append(int(incidents[i:i+H].any()))

    return np.array(X), np.array(y)

def train_model(X_train, y_train, model_type):
    if model_type == "log":
        model = LogisticRegression(max_iter = 1000, class_weight='balanced') 
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=89)
    else: 
        raise ValueError("Unknown model_type. Use 'log' or 'rf'.")
    
    model.fit(X_train, y_train) 
    return model

def evaluate(model, X_test, threshold):
    probs = model.predict_proba(X_test)[:,1]
    pred = (probs>threshold).astype(int)

    return probs, pred

def run_experiments(model, X_test, y_test, model_type):

    best_threshold = None
    best_score = 0

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        probs, pred = evaluate(model, X_test, threshold)

        recall = recall_score(y_test, pred)
        precision = precision_score(y_test, pred)
        score = 0.5*recall + 0.5*precision
        if score > best_score:
            best_score = score
            best_threshold = threshold

    probs, pred = evaluate(model, X_test, best_threshold)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    if model_type == "log": model_type = "Logistic Regression"
    elif model_type == "rf": model_type = "Random Forest"
    else: raise ValueError("Unknown model_type. Use 'log' or 'rf'.")

    print("\n")
    print(f"Best threshold ({model_type}):", best_threshold)
    print("F1-score (class 1):", f1)
    print("Recall (class 1)", recall)
    print("Precision (class 1)", precision)
    print("AUC:", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    main()