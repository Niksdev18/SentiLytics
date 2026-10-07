# baseline.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

train_df = train_df.dropna(subset=['text','label'])
val_df   = val_df.dropna(subset=['text','label'])
test_df  = test_df.dropna(subset=['text','label'])

X_train, y_train = train_df['text'].fillna(""), train_df['label']
X_val, y_val     = val_df['text'].fillna(""), val_df['label']
X_test, y_test   = test_df['text'].fillna(""), test_df['label']

print("Train class distribution:", Counter(y_train))

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# -----------------------------
# Feature extraction
# -----------------------------
tfidf_vec = TfidfVectorizer(max_features=7000, ngram_range=(1,3))
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_val_tfidf   = tfidf_vec.transform(X_val)
X_test_tfidf  = tfidf_vec.transform(X_test)

bow_vec = CountVectorizer(max_features=7000, ngram_range=(1,3))
X_train_bow = bow_vec.fit_transform(X_train)
X_test_bow  = bow_vec.transform(X_test)

# -----------------------------
# Handle class imbalance
# -----------------------------
smote = SMOTE(random_state=42)
X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train_enc)
X_train_bow_res, _ = smote.fit_resample(X_train_bow, y_train_enc)

# -----------------------------
# Define tuned base models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1500, C=1.5, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                           class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,subsample=0.8, 
                              colsample_bytree=0.8, eval_metric='mlogloss', random_state=42),
    "SVM": SVC(kernel='linear', C=2.0, probability=True, class_weight='balanced', random_state=42),
    "MNB": MultinomialNB()
}

metrics_list = []
stack_estimators = []

# -----------------------------
# Train base models & collect metrics
# -----------------------------
for name, model in models.items():
    if name == "MNB":
        model.fit(X_train_bow_res, y_train_res)
        y_pred = model.predict(X_test_bow)
    else:
        model.fit(X_train_tfidf_res, y_train_res)
        y_pred = model.predict(X_test_tfidf)
        stack_estimators.append((name, model))  # strong models for stacking

    y_pred_labels = le.inverse_transform(y_pred)
    acc = accuracy_score(y_test, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_labels, average='weighted')
    
    metrics_list.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    joblib.dump(model, f'models/{name}_model.pkl')

# -----------------------------
# Stacking ensemble
# -----------------------------
stacking = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42),
    cv=5,
    passthrough=True,
    stack_method='predict_proba'
)
stacking.fit(X_train_tfidf_res, y_train_res)
y_pred_stack = stacking.predict(X_test_tfidf)
y_pred_stack_labels = le.inverse_transform(y_pred_stack)

acc, precision, recall, f1, _ = accuracy_score(y_test, y_pred_stack_labels), *precision_recall_fscore_support(y_test, y_pred_stack_labels, average='weighted')
metrics_list.append({
    "Model": "StackingEnsemble",
    "Accuracy": acc,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
})
joblib.dump(stacking, 'models/stacking_ensemble_model.pkl')

# -----------------------------
# Weighted Soft Voting ensemble (include stacking + strong models)
# -----------------------------
voting_estimators = [
    ('Stacking', stacking),
    ('XGBoost', models['XGBoost']),
    ('LogisticRegression', models['LogisticRegression']),
    ('SVM', models['SVM'])
]

# Assign higher weight to stronger models based on validation performance
weights = [3, 3, 2, 2]  # stacking + XGBoost have more weight

voting_ensemble = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
    weights=weights
)
voting_ensemble.fit(X_train_tfidf_res, y_train_res)
y_pred_vote = voting_ensemble.predict(X_test_tfidf)
y_pred_vote_labels = le.inverse_transform(y_pred_vote)

acc, precision, recall, f1, _ = accuracy_score(y_test, y_pred_vote_labels), *precision_recall_fscore_support(y_test, y_pred_vote_labels, average='weighted')
metrics_list.append({
    "Model": "WeightedSoftVotingEnsemble",
    "Accuracy": acc,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
})
joblib.dump(voting_ensemble, 'models/weighted_soft_voting_model.pkl')

# -----------------------------
# Save metrics & vectorizers
# -----------------------------
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("models/model_metrics.csv", index=False)

joblib.dump(tfidf_vec, 'models/tfidf_vectorizer.pkl')
joblib.dump(bow_vec, 'models/bow_vectorizer.pkl')

print(" All base models, stacking and weighted soft voting ensemble trained successfully!")
