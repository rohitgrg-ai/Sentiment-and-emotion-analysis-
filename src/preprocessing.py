import pandas as pd
import re
import emoji


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


print("\n===== ORIGINAL vs CLEANED TEXT =====")

for i in range(5):
    print("\nOriginal:", df["text"].iloc[i])
    print("Cleaned :", df["clean_text"].iloc[i])