import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Load preprocessed dataset
df = pd.read_csv("data/processed_dataset.csv")


# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)


# Separate input and target variables
X_train_text = train_df["clean_text"]
X_test_text = test_df["clean_text"]

y_train_sentiment = train_df["sentiment"]
y_test_sentiment = test_df["sentiment"]

y_train_emotion = train_df["label"]
y_test_emotion = test_df["label"]


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)


# Learn vocabulary and calculate TF-IDF for training data
X_train = vectorizer.fit_transform(X_train_text)


# Transform testing data using the SAME vectorizer
X_test = vectorizer.transform(X_test_text)


print("\n===== TF-IDF INFORMATION =====")

print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])

print("Number of features :", X_train.shape[1])

print("Training matrix shape :", X_train.shape)
print("Testing matrix shape  :", X_test.shape)