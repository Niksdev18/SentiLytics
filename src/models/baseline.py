"""Classical ML Baseline Models (TF-IDF + LR/RF/XGB/SVM/MNB + Ensembles).

Reads 'text_classic' from data/processed/train.csv, val.csv, and test.csv.
Trains tuned classifiers and logs evaluation metrics to models/model_metrics.csv.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_baselines(data_dir: Optional[Path] = None, models_dir: Optional[Path] = None) -> pd.DataFrame:
    """Train all baseline classical models and save artifacts and metrics.

    Args:
        data_dir: Path to directory containing train.csv, val.csv, test.csv.
        models_dir: Path to directory to store trained .pkl models and metrics.

    Returns:
        pd.DataFrame containing comparison metrics.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if data_dir is None:
        data_dir = root / "data" / "processed"
    if models_dir is None:
        models_dir = root / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Processed datasets not found in {data_dir}. Run split_data.py first.")

    train_df = pd.read_csv(train_path).dropna(subset=["text_classic", "label"])
    test_df = pd.read_csv(test_path).dropna(subset=["text_classic", "label"])

    X_train = train_df["text_classic"].astype(str)
    y_train = train_df["label"].astype(str)

    X_test = test_df["text_classic"].astype(str)
    y_test = test_df["label"].astype(str)

    logger.info(f"Train class distribution: {Counter(y_train)}")

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    _ = le.transform(y_test)
    joblib.dump(le, models_dir / "baseline_label_encoder.pkl")

    # Feature extraction (TF-IDF & Count)
    tfidf_vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 3))
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)

    bow_vec = CountVectorizer(max_features=7000, ngram_range=(1, 3))
    X_train_bow = bow_vec.fit_transform(X_train)
    X_test_bow = bow_vec.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train_enc)
    X_train_bow_res, _ = smote.fit_resample(X_train_bow, y_train_enc)

    # Define base models
    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=1500, C=1.5, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42
        ),
        "SVM": SVC(kernel="linear", C=2.0, probability=True, class_weight="balanced", random_state=42),
        "MNB": MultinomialNB(),
    }

    metrics_list: List[Dict[str, Any]] = []
    stack_estimators = []

    for name, model in base_models.items():
        logger.info(f"Training {name}...")
        if name == "MNB":
            model.fit(X_train_bow_res, y_train_res)
            y_pred = model.predict(X_test_bow)
        else:
            model.fit(X_train_tfidf_res, y_train_res)
            y_pred = model.predict(X_test_tfidf)
            stack_estimators.append((name, model))

        y_pred_labels = le.inverse_transform(y_pred)
        acc = accuracy_score(y_test, y_pred_labels)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_labels, average="weighted", zero_division=0)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        })
        joblib.dump(model, models_dir / f"{name}_model.pkl")

    # Stacking ensemble
    logger.info("Training Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        cv=5,
        passthrough=True,
        stack_method="predict_proba",
    )
    stacking.fit(X_train_tfidf_res, y_train_res)
    y_pred_stack = stacking.predict(X_test_tfidf)
    y_pred_stack_labels = le.inverse_transform(y_pred_stack)
    acc_stack = accuracy_score(y_test, y_pred_stack_labels)
    prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
        y_test, y_pred_stack_labels, average="weighted", zero_division=0
    )
    metrics_list.append({
        "Model": "StackingEnsemble",
        "Accuracy": round(acc_stack, 4),
        "Precision": round(prec_s, 4),
        "Recall": round(rec_s, 4),
        "F1": round(f1_s, 4),
    })
    joblib.dump(stacking, models_dir / "stacking_ensemble_model.pkl")

    # Soft Voting ensemble
    logger.info("Training Soft Voting Ensemble...")
    voting_estimators = [
        ("Stacking", stacking),
        ("XGBoost", base_models["XGBoost"]),
        ("LogisticRegression", base_models["LogisticRegression"]),
        ("SVM", base_models["SVM"]),
    ]
    voting = VotingClassifier(estimators=voting_estimators, voting="soft", weights=[3, 3, 2, 2])
    voting.fit(X_train_tfidf_res, y_train_res)
    y_pred_vote = voting.predict(X_test_tfidf)
    y_pred_vote_labels = le.inverse_transform(y_pred_vote)
    acc_vote = accuracy_score(y_test, y_pred_vote_labels)
    prec_v, rec_v, f1_v, _ = precision_recall_fscore_support(
        y_test, y_pred_vote_labels, average="weighted", zero_division=0
    )
    metrics_list.append({
        "Model": "WeightedSoftVotingEnsemble",
        "Accuracy": round(acc_vote, 4),
        "Precision": round(prec_v, 4),
        "Recall": round(rec_v, 4),
        "F1": round(f1_v, 4),
    })
    joblib.dump(voting, models_dir / "weighted_soft_voting_model.pkl")

    # Save vectorizers and metrics table
    joblib.dump(tfidf_vec, models_dir / "tfidf_vectorizer.pkl")
    joblib.dump(bow_vec, models_dir / "bow_vectorizer.pkl")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(models_dir / "model_metrics.csv", index=False)
    logger.info(f"Saved metrics to {models_dir / 'model_metrics.csv'}")

    return metrics_df


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    df = train_baselines()
    print("\nBaseline Training Results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
