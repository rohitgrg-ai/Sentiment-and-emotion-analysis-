import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("data/text.csv")


# ==================================================
# 1. BASIC DATASET INFORMATION
# ==================================================

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFORMATION =====")
print(df.info())


# ==================================================
# 2. MISSING VALUES
# ==================================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== MISSING VALUE PERCENTAGE =====")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)


# Visualize missing values
plt.figure(figsize=(10, 5))

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    sns.barplot(x=missing.index, y=missing.values)
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found.")


# ==================================================
# 3. DUPLICATE ROWS
# ==================================================

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== DUPLICATE ROW PERCENTAGE =====")
duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
print(f"{duplicate_percentage:.2f}%")


# ==================================================
# 4. UNIQUE VALUES
# ==================================================

print("\n===== UNIQUE SENTIMENTS =====")
print(df["sentiment"].unique())

print("\n===== UNIQUE EMOTION LABELS =====")
print(sorted(df["label"].unique()))

print("\n===== NUMBER OF UNIQUE TEXTS =====")
print(df["text"].nunique())


# ==================================================
# 5. SENTIMENT DISTRIBUTION
# ==================================================

print("\n===== SENTIMENT DISTRIBUTION =====")
print(df["sentiment"].value_counts())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=df["sentiment"].value_counts().index
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==================================================
# 6. EMOTION DISTRIBUTION
# ==================================================

print("\n===== EMOTION DISTRIBUTION =====")
print(df["label"].value_counts().sort_index())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="label",
    order=sorted(df["label"].unique())
)

plt.title("Emotion Label Distribution")
plt.xlabel("Emotion Label")
plt.ylabel("Number of Texts")
plt.tight_layout()
plt.show()


# ==================================================
# 7. SENTIMENT + EMOTION DISTRIBUTION
# ==================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="label",
    hue="sentiment"
)

plt.title("Emotion Distribution by Sentiment")
plt.xlabel("Emotion Label")
plt.ylabel("Number of Texts")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()


# ==================================================
# 8. TEXT LENGTH ANALYSIS
# ==================================================

df["text_length"] = df["text"].astype(str).str.len()

print("\n===== TEXT LENGTH STATISTICS =====")
print(df["text_length"].describe())


# Histogram

plt.figure(figsize=(10, 5))

sns.histplot(
    df["text_length"],
    bins=50,
    kde=True
)

plt.title("Distribution of Text Length")
plt.xlabel("Text Length (Characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Boxplot

plt.figure(figsize=(10, 4))

sns.boxplot(
    x=df["text_length"]
)

plt.title("Boxplot of Text Length")
plt.xlabel("Text Length (Characters)")
plt.tight_layout()
plt.show()


# ==================================================
# 9. TEXT LENGTH BY EMOTION
# ==================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="label",
    y="text_length"
)

plt.title("Text Length Distribution by Emotion")
plt.xlabel("Emotion Label")
plt.ylabel("Text Length")
plt.tight_layout()
plt.show()


# ==================================================
# 10. TEXT LENGTH BY SENTIMENT
# ==================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="sentiment",
    y="text_length"
)

plt.title("Text Length Distribution by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Text Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==================================================
# 11. CREATE TEXT LENGTH CATEGORIES
# ==================================================

def categorize_length(length):

    if length <= 50:
        return "Short"

    elif length <= 100:
        return "Medium"

    elif length <= 200:
        return "Long"

    else:
        return "Very Long"


df["text_length_category"] = df["text_length"].apply(
    categorize_length
)


print("\n===== TEXT LENGTH CATEGORIES =====")
print(df["text_length_category"].value_counts())


# Plot categories

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="text_length_category",
    order=["Short", "Medium", "Long", "Very Long"]
)

plt.title("Text Length Categories")
plt.xlabel("Text Length Category")
plt.ylabel("Number of Texts")
plt.tight_layout()
plt.show()


# ==================================================
# 12. LENGTH CATEGORY vs EMOTION
# ==================================================

length_emotion = pd.crosstab(
    df["text_length_category"],
    df["label"]
)

print("\n===== TEXT LENGTH CATEGORY × EMOTION =====")
print(length_emotion)


plt.figure(figsize=(10, 6))

sns.heatmap(
    length_emotion,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Text Length Category vs Emotion")
plt.xlabel("Emotion Label")
plt.ylabel("Text Length Category")
plt.tight_layout()
plt.show()


# ==================================================
# 13. SENTIMENT × EMOTION CONTINGENCY TABLE
# ==================================================

sentiment_emotion = pd.crosstab(
    df["sentiment"],
    df["label"]
)

print("\n===== SENTIMENT × EMOTION =====")
print(sentiment_emotion)


# ==================================================
# 14. HEATMAP: SENTIMENT vs EMOTION
# ==================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    sentiment_emotion,
    annot=True,
    fmt="d",
    cmap="YlGnBu"
)

plt.title("Sentiment vs Emotion Heatmap")
plt.xlabel("Emotion Label")
plt.ylabel("Sentiment")
plt.tight_layout()
plt.show()


# ==================================================
# 15. PERCENTAGE HEATMAP
# ==================================================

sentiment_emotion_percent = pd.crosstab(
    df["sentiment"],
    df["label"],
    normalize="index"
) * 100

print("\n===== SENTIMENT × EMOTION (%) =====")
print(sentiment_emotion_percent.round(2))


plt.figure(figsize=(10, 6))

sns.heatmap(
    sentiment_emotion_percent,
    annot=True,
    fmt=".2f",
    cmap="Oranges"
)

plt.title("Sentiment vs Emotion (%)")
plt.xlabel("Emotion Label")
plt.ylabel("Sentiment")
plt.tight_layout()
plt.show()


# ==================================================
# 16. CATEGORICAL ENCODING
# ==================================================

# Convert categorical variables into numerical values

df["sentiment_encoded"] = pd.Categorical(
    df["sentiment"]
).codes

df["length_category_encoded"] = pd.Categorical(
    df["text_length_category"],
    categories=["Short", "Medium", "Long", "Very Long"],
    ordered=True
).codes


# ==================================================
# 17. CORRELATION DATAFRAME
# ==================================================

correlation_data = df[
    [
        "label",
        "sentiment_encoded",
        "text_length",
        "length_category_encoded"
    ]
]

print("\n===== CORRELATION MATRIX =====")
print(correlation_data.corr())


# ==================================================
# 18. CORRELATION HEATMAP
# ==================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Encoded Features")
plt.tight_layout()
plt.show()


# ==================================================
# 19. EMOTION vs TEXT LENGTH CATEGORY
# ==================================================

emotion_length_percentage = pd.crosstab(
    df["label"],
    df["text_length_category"],
    normalize="index"
) * 100

plt.figure(figsize=(10, 6))

sns.heatmap(
    emotion_length_percentage,
    annot=True,
    fmt=".1f",
    cmap="Purples"
)

plt.title("Emotion vs Text Length Category (%)")
plt.xlabel("Text Length Category")
plt.ylabel("Emotion Label")
plt.tight_layout()
plt.show()


# ==================================================
# 20. VIOLIN PLOT
# ==================================================

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df,
    x="label",
    y="text_length"
)

plt.title("Text Length Distribution Across Emotions")
plt.xlabel("Emotion Label")
plt.ylabel("Text Length")
plt.tight_layout()
plt.show()


# ==================================================
# 21. EMPTY TEXT ANALYSIS
# ==================================================

empty_text = (
    df["text"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("\n===== EMPTY TEXT =====")
print(empty_text)


# ==================================================
# 22. TEXT DUPLICATES
# ==================================================

print("\n===== TEXT DUPLICATES =====")

text_duplicates = df["text"].duplicated().sum()

print("Duplicate text entries:", text_duplicates)

print(
    "Duplicate percentage:",
    f"{(text_duplicates / len(df)) * 100:.2f}%"
)


# ==================================================
# 23. DUPLICATE TEXT LABEL CONSISTENCY
# ==================================================

duplicate_texts = df[
    df["text"].duplicated(keep=False)
]

print("\n===== DUPLICATE TEXT LABEL CONSISTENCY =====")

print(
    "Duplicate rows:",
    len(duplicate_texts)
)

print(
    "Unique duplicate texts:",
    duplicate_texts["text"].nunique()
)


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

print(
    "Duplicate texts with different labels:",
    conflicting.shape[0]
)


# ==================================================
# 24. LABEL DISTRIBUTION TABLE
# ==================================================

label_summary = (
    df.groupby("label")
    .agg(
        samples=("label", "size"),
        avg_text_length=("text_length", "mean"),
        min_text_length=("text_length", "min"),
        max_text_length=("text_length", "max")
    )
    .round(2)
)

print("\n===== LABEL SUMMARY =====")
print(label_summary)


# ==================================================
# 25. SENTIMENT SUMMARY TABLE
# ==================================================

sentiment_summary = (
    df.groupby("sentiment")
    .agg(
        samples=("sentiment", "size"),
        avg_text_length=("text_length", "mean"),
        min_text_length=("text_length", "min"),
        max_text_length=("text_length", "max")
    )
    .round(2)
)

print("\n===== SENTIMENT SUMMARY =====")
print(sentiment_summary)