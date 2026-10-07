import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads (only first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True, lemmatize=True):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters (keep $ % for finance)
    text = re.sub(r"[^a-zA-Z$% ]", " ", text)

    # 5. Normalize financial terms
    text = text.replace("us$", "usd").replace("$", " usd ")

    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stopwords (optional)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # 8. Lemmatization
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Join back
    text = " ".join(tokens)

    return text


# Example usage
df = pd.read_csv("data/raw/financial_phrasebank_clean.csv")
df['cleaned_text'] = df['sentence'].apply(lambda x: clean_text(str(x)))
df.drop_duplicates(subset=['cleaned_text'], inplace=True)

df.to_csv("data/processed/financial_news_cleaned.csv", index=False)
print(df[['sentence', 'cleaned_text']].head())
