import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_most_freq_words():
	df = pd.read_csv('./output/cleaned_data.csv')

	# Count the frequency of each word
	word_counts = Counter(df['content_processed_stop'].tolist())

	# Create a bar chart
	plt.bar(word_counts.keys(), word_counts.values())
	plt.xlabel('Words')
	plt.ylabel('Frequency')
	plt.title('Most Frequent Words in Column')

	# Show the plot
	st.pyplot(plt.show())

#plot_most_freq_words()