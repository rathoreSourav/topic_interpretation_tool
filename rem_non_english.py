import pandas as pd
import string

# Define a function to check if a word is in English
def is_english(word):
    try:
        word.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# Define a function to clean the data in the column
def clean_column(row):
    cleaned_row = []
    for item in row:
        words = item.split()
        cleaned_words = [str(word) for word in words if is_english(str(word))]
        cleaned_item = " ".join(cleaned_words)
        cleaned_row.append(cleaned_item)
    return cleaned_row


    # # Remove punctuation and convert to lowercase
    # column = column.str.translate(str.maketrans("", "", string.punctuation)).str.lower()
    # # Split the column into words
    # words = column.str.split()
    # # Keep only the English words
    # print(words)
    # words = [str(word) for word_list in words for word in word_list if is_english(str(word))]
    # # Join the words back together
    # column = " ".join(words)
    # return column

# # Clean the data in the column
# df["column_name"] = clean_column(df["column_name"])

# # Save the cleaned data back to the CSV file
# df.to_csv("cleaned_file.csv", index=False)
