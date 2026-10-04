import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Convert emojis to text descriptions
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load dataset
df = pd.read_csv("data/text2.csv")

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"])
df["clean_text"] = df["text"].apply(clean_text)
df.to_csv("data/processed_dataset.csv", index=False)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["sentiment"]
)

print("\n===== DATASET SPLIT =====")

print("Total samples :", len(df))
print("Training      :", len(train_df))
print("Testing       :", len(test_df))


print("\n===== TRAINING SENTIMENT =====")
print(train_df["sentiment"].value_counts())


print("\n===== TESTING SENTIMENT =====")
print(test_df["sentiment"].value_counts())


print("\n===== TRAINING EMOTION =====")
print(train_df["label"].value_counts().sort_index())


print("\n===== TESTING EMOTION =====")
print(test_df["label"].value_counts().sort_index())