import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset.csv", encoding="latin1")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Apply preprocessing on article_content
df['clean_text'] = df['article_content'].apply(preprocess_text)

# Verify preprocessing
print("----- ORIGINAL TEXT -----")
print(df['article_content'].iloc[0][:300])

print("\n----- CLEANED TEXT -----")
print(df['clean_text'].iloc[0][:300])

# Save cleaned dataset (IMPORTANT for Day 3)
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_dataset.csv")
