import csv
import csv
import nltk
from nltk.corpus import stopwords
import pandas as pd
import string

# nltk.download("stopwords")
# stop_words = stopwords.words('english')

def read_stop_words(filename):
    with open(filename, "r") as file:
        return set(word.strip().lower() for word in file.readlines())



def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    print('text is: ', preprocess_text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = read_stop_words("test.txt")
    text = [word for word in text.split() if len(word) > 4 and word not in stop_words]

    return text

def run_preprocess():
    df = pd.read_csv("cleaned_doc.csv")
    print("I am here")
    df['content_processed_stop'] = df['content_processed_stops'].apply(lambda x: preprocess_text(x))    
    df.to_csv("output.csv", index=False)


#run_preprocess()

# # def rm_stopwords():
# #     read = pd.read_csv("cleaned_doc.csv")
# #     print(read['content_processed_lower'].head())
# #     read['new']  = \
# #     read['content_processed_lower'].apply(lambda x: "".join([word for word in x.split() if word.lower() not in (stop_words)]))

# def rm_stopwords():
#     # //nltk.download("stopwords")
#     # //stop_words = set(stopwords.words("english"))
#     # //print("stopwrods type:",type(stop_words))
#     stop_words = read_stop_words("test.txt")
   
#     #stop_words = stop_words.extend(read_stop_words("healthcare_stopwords.txt"))
#     #print("stopwrods type:",type(stop_words))

#     df = pd.read_csv("cleaned_doc.csv")
#     #lambda x: " ".join([word for word in x.split() if len(word) >= 4 and word.strip().lower() not in (stop_words)])

#     df["content_processed_stop"] = \
#     df["content_processed_stops"].apply(lambda x: " ".join([word for word in x.split() if len(word) >= 4 and word.strip().lower() not in (stop_words)]))
#     df.to_csv("output.csv", index=False)
#  