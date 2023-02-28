__author__      = "Sourav Singh"
__version__ = "1.0.1"
__project__ = "Topic interpretation tool"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)

def plotTF_IDF():
	df = pd.read_csv('./output/extracted_doc.csv')
	st.header('TF-IDF Score')
	st.markdown("---")
	st.subheader("Most discussed topics in the documents")
	# Preprocess the documents
	tfidf = TfidfVectorizer(stop_words='english', max_df=len(df['Keywords']), min_df=1, lowercase=True)
	tfidf_matrix = tfidf.fit_transform(df["Keywords"])
	# Compute the TF-IDF scores
	tfidf_scores = tfidf_matrix.sum(axis=0).A1
	tfidf_words = tfidf.get_feature_names()
	# Plot the bar graph
	#Create horizontal bar chart
	fig, ax = plt.subplots()
	ax.barh(tfidf_words[:50], tfidf_scores[:50], height=.8)
	
	# Adjust spacing between bars
	#ax.set_ylim(-1, len(tfidf_words[:50])-1)
	ax.set_yticks(np.arange(len(tfidf_words[:50])))
	ax.set_yticklabels(tfidf_words[:50], fontsize=6)
	#ax.set_yticklabels(tfidf_words[:20], fontsize=12) 
	plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.05, hspace=0.5)
	
	# Add labels to the plot
	ax.tick_params(axis='y', labelrotation=0)

	plt.xlabel("TF-IDF Score")
	plt.ylabel("Most talked topics in the document")
	# Show the plot
	plt.savefig('plotTF_IDF.png')
	st.pyplot(plt.show())
	st.markdown("---")
	st.markdown("---")


def plot_companies():
	try:
		df = pd.read_csv('./output/extracted_doc.csv')
		st.header('Topic Label Grouped by Companies')
		st.markdown("---")
		#legends = set(df['Topic Label'])
		# Extract the unique values from a column
		legends = df['Topic Label'].unique().tolist()

		# Keep the rows ordered by preserving the order of the unique values
		legends = [value for value in df['Topic Label'] if value in legends]

		unique_legends = []
		for item in legends:
		    if item not in unique_legends:
		        unique_legends.append(item)

		#st.write("legends:", unique_legends)

		# Group the data by Group and Company
		grouped = df.groupby(['Topic_Num', 'Name']).size().reset_index(name='Count')
		ind = 0
		# Plot the data for each group
		for group, subset in grouped.groupby('Topic_Num'):
			#st.subheader(unique_legends[ind])
			ax = subset.plot(x='Name', y='Count', kind='bar')
			plt.title(f'Topic: {unique_legends[ind]}', fontweight="bold")
			plt.xlabel('Company')
			plt.ylabel('Occurence')
			st.pyplot(plt.show(), figsize=(2, 2))
			st.markdown("---")
			ind = ind+1
	except:
		st.write('Please run the label generator to visulaize company distribution per topic!')		
			

def insights_visulaize():
	plotTF_IDF()
	plot_companies()	

#insights_visulaize()	