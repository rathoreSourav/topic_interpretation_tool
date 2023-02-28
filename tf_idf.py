import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def plotTF_IDF():
	# Load the CSV file into a DataFrame
	df = pd.read_csv("extracted_doc.csv")

	# Preprocess the documents
	tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, lowercase=True)
	tfidf_matrix = tfidf.fit_transform(df["Keywords"])

	# Compute the TF-IDF scores
	tfidf_scores = tfidf_matrix.sum(axis=0).A1
	tfidf_words = tfidf.get_feature_names()

	# Plot the bar graph
	plt.barh(tfidf_words[:50], tfidf_scores[:50])

	# Add labels to the plot
	plt.xlabel("TF-IDF Score")
	plt.ylabel("Most talked topics in the document")

	# Show the plot
	plt.show()

plotTF_IDF()
