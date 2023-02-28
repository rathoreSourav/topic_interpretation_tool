import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]

    return text

df['text_column'] = df['text_column'].apply(lambda x: preprocess_text(x))