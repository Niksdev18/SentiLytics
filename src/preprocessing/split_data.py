import pandas as pd
from sklearn.model_selection import train_test_split

# Load your cleaned dataset
df = pd.read_csv("data/processed/financial_news_cleaned.csv")

# Features (X) and labels (y)
X = df['cleaned_text']
y = df['label']   # <-- replace with your actual label column name

# First split: Train (70%) vs Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: Validation (15%) vs Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,   # half of 30% → 15%
    random_state=42,
    stratify=y_temp
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# ✅ Optional: Save to files (for reproducibility)
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df   = pd.DataFrame({'text': X_val, 'label': y_val})
test_df  = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
