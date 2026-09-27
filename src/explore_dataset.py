import pandas as pd

# Load dataset
df = pd.read_csv("data/text.csv")

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== SENTIMENT DISTRIBUTION =====")
print(df["sentiment"].value_counts())

print("\n===== EMOTION DISTRIBUTION =====")
print(df["label"].value_counts().sort_index())

print("\n===== UNIQUE SENTIMENTS =====")
print(df["sentiment"].unique())

print("\n===== UNIQUE EMOTION LABELS =====")
print(sorted(df["label"].unique()))

print("\n===== EMPTY TEXT =====")
print((df["text"].str.strip() == "").sum())

print("\n===== TEXT DUPLICATES =====")
print(df["text"].duplicated().sum())

print("\n===== TEXT LENGTH =====")
print(df["text"].str.len().describe())

print("\n===== DUPLICATE TEXT LABEL CONSISTENCY =====")

duplicate_texts = df[df["text"].duplicated(keep=False)]

print("Duplicate rows:", len(duplicate_texts))

print("\nNumber of unique duplicate texts:")
print(duplicate_texts["text"].nunique())

print("\nDuplicate texts with different labels:")

conflicting = (
    duplicate_texts
    .groupby("text")
    .agg(
        emotion_classes=("label", "nunique"),
        sentiment_classes=("sentiment", "nunique")
    )
)

conflicting = conflicting[
    (conflicting["emotion_classes"] > 1) |
    (conflicting["sentiment_classes"] > 1)
]

print(conflicting.shape[0])