__author__      = "Sourav Singh"
__version__ = "1.0.1"

import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer
from pprint import pprint
import pandas as pd
from itertools import chain
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import summ
#gensim summarize
from gensim.summarization import summarize
#bert summarize
from summarizer import Summarizer
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from random import sample
from typing import List
import csv
import color_coding as cc
from bertLabel import generate_label

max_articles_for_label = 5

w_tokenizer = WhitespaceTokenizer()
#lemmatization
lemma = WordNetLemmatizer()

def write_data(data):
	header = ["topic_num", "keywords", "summary"]
	with open('./output/summary_data.csv', 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(header)
		for row in data:
			writer.writerow(row)  

#count number of data in a topic
def count_till_value_changes(column, start_index):
	start_index = start_index
	current_value = column[start_index]
	st.write('current column topic nbumber: ', current_value)
	count = 0
	for value in column.iloc[start_index:].values:
		st.write('value: ' ,value)
		try:
			int(value)
			if value == current_value:
	  			count += 1
	            #st.write('value: ' ,value)
			else:
				break
		except:
			count += 1	
			continue	
	st.write('total topic count: ', count)
	return count

#break the sentence into words
def convert(csv_data):
	splitted_sent_list = []
	for sent in csv_data:
		splitted_temp_sent_list = []
		split_one = ' '.join(sent).split( )
		split_two = ' '.join(split_one).split( )
		splitted_temp_sent_list += split_two
		splitted_sent_list.append(splitted_temp_sent_list)
	return splitted_sent_list


#tokenize the list data 
def tokenize_list_data(processed_list):
	st.write("tokenizing the corpus data...")
	tokenizer = RegexpTokenizer(r'\w+')
	#print(tokenizer)
	processed_txt = []
	for r in processed_list:
		yield tokenizer.tokenize(r)


#tokenize the dataframe data
def tokenized_data(csv_data):
	st.write("tokenizing the corpus data...")
	tokenizer = RegexpTokenizer(r'\w+')
	for r in csv_data['text_lemmatized']:
		processed_txt+=str(str(lemma.lemmatize(r, pos="a")))
		#print(type(processed_txt))
		yield tokenizer.tokenize(str(processed_txt))


def extract_document(csv_data, keywords):
	st.write("extracting matching documents...")
	# extract sentences containing keywords
	fileinF = csv_data['content']
	for sent in fileinF:
		tokenized_sent = [word.lower() for word in w_tokenizer.tokenize(sent)]
		if any(keyw in tokenized_sent for keyw in keywords):
			fileinF.append(sent)
	#print(fileinF)

	
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
	#st.write("computing model coherence...")
	coherence_values = []
	model_list = []
	for num_topics in range(start, limit, step):
		model = LdaModel(corpus, num_topics, dictionary)
		model_list.append(model)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
		coherence_values.append(coherencemodel.get_coherence())

	return model_list, coherence_values


def show_coherence_graph(coherence_values, limit, start, step):
	st.set_option('deprecation.showPyplotGlobalUse', False)	
	st.write("")
	st.subheader("LDA MODEL COHERENCE GRAPH")
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
	#fig = plt.Figure(figsize=(12,7))
	st.pyplot(plt.show())
	#print(plt.show())


#for extracting documents from data frame
def top_sent(ldamodel, corpus, texts, name):
	sent_topics_df = pd.DataFrame()
	for i, row in enumerate(ldamodel[corpus]):
		row = sorted(row, key=lambda x: (x[1]), reverse=True)
		# Get the Dominant topic, Perc Contribution and Keywords for each document
		for j, (topic_num, prop_topic) in enumerate(row):
			if j == 0:  # => dominant topic
				wp = ldamodel.show_topic(topic_num)
				topic_keywords = ", ".join([word for word, prop in wp])
				sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
			else:
				break
	sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text and name to the end of the output
	contents = pd.Series(texts)
	name = pd.Series(name)
	sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
	sent_topics_df = pd.concat([sent_topics_df, name], axis=1)
	return(sent_topics_df)


def get_doc_for_topics(csv_data, ldamodel, corpus, doc_cut_off):
	doc_cut_off_local = doc_cut_off
	df_topic_sents_keywords = top_sent(ldamodel=ldamodel, 
										corpus=corpus,
										texts=csv_data['content'],
										name=csv_data['name'])
	# Formatting
	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Name']

	sent_topics_sorteddf_mallet = pd.DataFrame()

	sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

	#increase head to see more dcouments there
	for i, grp in sent_topics_outdf_grpd:
	    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
	                                             grp.sort_values(['Perc_Contribution'], ascending=[0])], 
	                                            axis=0)

	#print(sent_topics_sorteddf_mallet['Keywords'])

	# Reset Index    
	sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

	# Format
	sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text", "Name"]

	#cut off documents
	sent_topics_sorteddf_mallet = sent_topics_sorteddf_mallet[sent_topics_sorteddf_mallet['Topic_Perc_Contrib'] > doc_cut_off_local]

	sent_topics_sorteddf_mallet.to_csv('./output/extracted_doc.csv', header=True, index=False)
	st.write(sent_topics_sorteddf_mallet)


	#summary_extract()
	
	print("-----EO get_topic_doucments")

def label_generate():
	#model = Summarizer()
	#model = Summarizer(model='distilbert-base-uncased')
	print("auto label generation")
	datafile = pd.read_csv("./output/extracted_doc.csv")
	filename = list(datafile['Text'])
	topic_number = datafile['Topic_Num']
	summary_list=[]
	text=''
	count=0
	slist=[]
	topic_num = 0
	start_row = 0
	end_row = 0
	#end_row = doc_cut_off_local
	while end_row < len(topic_number) and topic_num<=int(st.session_state['value']):
		end_row =  end_row + count_till_value_changes(topic_number, start_index=start_row)
		print('start row: ',start_row)
		print('end row: ',end_row)
		for i in filename:
			count+= 1
			summary_list.append(i)
			text += i
			slist.append(i)
			#todo add only text from each company
			if(count == end_row):
				print("documents for the topic")
				topicList= datafile['Keywords'].unique()[topic_num]
				st.write("Topic ", topic_num)
				st.write("Topic Keywrods: ", topicList)
				topic_label_gen = generate_label('', slist)
				st.write("Topic Label: ", generate_label('', slist))
				datafile.loc[start_row:end_row, 'Topic Label'] = topic_label_gen
				text=''
				count=0
				topic_num=topic_num+1
				slist=[]
				start_row = end_row

	datafile.to_csv('./output/extracted_doc.csv', index=False)


#todo add only text from each company
def summary_extract():
	#model = Summarizer()
	model = Summarizer(model='distilbert-base-uncased')
	print("extracting summary of documents")
	datafile = pd.read_csv("./output/extracted_doc.csv")
	filename = list(datafile['Text'])
	topic_number = datafile['Topic_Num']
	summary_list=[]
	text=''
	count=0
	slist=[]
	topic_num = 0
	start_row = 0
	end_row = 0
	#end_row = doc_cut_off_local
	while end_row < len(topic_number) and topic_num<=int(7):
		end_row =  end_row + count_till_value_changes(topic_number, start_index=start_row)
		st.write('start row before: ',start_row)
		st.write('end row before: ',end_row)
		for i in filename:
			count+= 1
			summary_list.append(i)
			text += i
			slist.append(i)
			if(count == end_row):
				print("documents for the topic")
				topicList= datafile['Keywords'].unique()[topic_num]
				st.write("Topic ", topic_num)
				st.write("Topic Keywrods: ", topicList)
				st.write("Number of documents summarized: ", len(slist))
				st.write("Summariy: ")
				text=''
				count=0
				topic_num=topic_num+1
				slist=[]
				start_row = end_row
				end_row =  end_row + count_till_value_changes(topic_number, start_index=start_row) 
	

#updated summary methods dynamic
def summary_extract_updated():
	#st.session_state['value'] = '8'
	print("extracting summary of documents")
	df = pd.read_csv("./output/extracted_doc.csv")
	st.write(df.head(5))
	data = []
	for i in range(int(st.session_state['value'])):
		#st.write('i is :', i)
		df_selected = df[df['Topic_Num'] == i]
		df_selected = df_selected.drop_duplicates(subset=['Name']).head(10)
		#st.write(df_selected.head(5))
		st.write('Total number of documents under topic ', i ,' are: ', len(df_selected))
		artilce = ''
		filename = list(df_selected['Text'])
		for text in filename:
			artilce += text
		try:	
			summary = summarize(artilce, word_count=400)
		except:
			summary = "Exception occured while summarization"
		topicList= df_selected['Keywords'].unique()[0]
		st.write("Topic ", i)
		st.write("Topic Keywrods: ", topicList)
		st.write("Summariy: ", summary)
		data.append([i, topicList, summary])
		write_data(data)


def auto_labelling_generation():
	# Change background color
	st.markdown(
	    """
	    <h2>
	    	Generating topic labels!
	    </h2>
	    """,
	    unsafe_allow_html=True
	)
	#st.session_state['value'] = '8'
	print("auto_labelling_generation method")
	df = pd.read_csv("./output/extracted_doc.csv")
	start_row = 0
	end_row = 0
	for i in range(int(st.session_state['value'])):
		df_selected = df[df['Topic_Num'] == i]
		top_5_unique_rows = df_selected.drop_duplicates(subset=['Name']).head(10)
		#st.write(top_5_unique_rows)
		artilce_list = []
		filename = list(top_5_unique_rows['Text'])


		#count_articles = 0
		for text in filename:
		#	if count_articles == 5:
		#		break
			artilce_list.append(text)
		#	count_articles += 1
		#try:	
		topic_label_gen = generate_label('', artilce_list)	
		#except:
		#	topic_label_gen = "Exception occured while generating labels"

		#delete artciles from previous topic
		artilce_list=[]
		st.write("Topic ", i)
		st.write("Topic Keywrods: ", df_selected['Keywords'].unique()[0])
		st.write("Topic label: ", topic_label_gen)
		start_row = end_row
		end_row = end_row + len(df_selected)
		df.loc[start_row:end_row, 'Topic Label'] = topic_label_gen

	#save to same file	
	df.to_csv('./output/extracted_doc.csv', index=False)	




def topic_modelling(csv_data, num_topics):
	print("topic_modelling method of model_topic.py called")
	tokens_list= convert(csv_data['text_lemmatized'])
	tokens = tokenize_list_data(tokens_list)
	id2word = corpora.Dictionary(tokens_list)
	st.session_state['id2word'] = id2word
	corpus = [id2word.doc2bow(text) for text in tokens_list]
	#print(corpus)
	st.session_state['corpus'] = corpus
	# Build LDA model
	lda_model = LdaModel(corpus=corpus, 
							id2word=id2word,
							num_topics=num_topics,
							passes=50,
							iterations=100,
							chunksize = 3000, 
							eval_every = None, 
							random_state=0
							)
	
	#Print the Keyword in the topics
	st.subheader("Topic Keywords")
	st.write()
	#st.table(lda_model.print_topics())

	# Define table headers
	headers = ['Topic Number', 'Topic words']

	# Define table rows
	rows = []

	# Print table with headers and rows
	#Todo remove , from topic terms
	st.write('<h2>Table Example</h2>', unsafe_allow_html=True)
	num =1
	for topiceach in lda_model.print_topics():
		cc.get_up_color(topiceach)
		# Define table rows
		rows.append([num , cc.get_up_color(topiceach)])
		num += 1

	st.write('<table><tr>{}</tr>{}</table>'.format(
    ''.join('<th>{}</th>'.format(h) for h in headers),
    ''.join('<tr>{}</tr>'.format(''.join('<td>{}</td>'.format(val) for val in row)) for row in rows)),
    unsafe_allow_html=True)

	st.session_state['lda_model'] = lda_model


def get_coherence_score(csv_data):
	tokens_list= convert(csv_data['text_lemmatized'])
	tokens = tokenize_list_data(tokens_list)
	id2word = corpora.Dictionary(tokens_list)
	corpus = [id2word.doc2bow(text) for text in tokens_list]
	#get model list and coherence values to plot the graph
	return compute_coherence_values(dictionary=id2word, corpus=corpus, texts=csv_data['text_lemmatized'], start=5, limit=50, step=5)


def visulize_result():
	try:
		vis = pyLDAvis.gensim.prepare(topic_model=st.session_state['lda_model'], 
										corpus=st.session_state['corpus'], 
										dictionary=st.session_state['id2word'])
		pyLDAvis.save_html(data = vis, fileobj='res.html')
		HtmlFile = open("res.html", 'r', encoding='utf-8')
		source_code = HtmlFile.read() 
		components.html(source_code, width=1200, height=1000)

	except:
		st.write("something went wrong, make sure you have run the topic model at least once from the home page")

#not in use, please see the #get_coherence_score 
def get_optimal_model(csv_data):
	tokens_list= convert(csv_data['text_lemmatized'])
	#print(tokens_list)

	tokens = tokenize_list_data(tokens_list)
	#print(tokens)

	id2word = corpora.Dictionary(tokens_list)
	#print(id2word)

	corpus = [id2word.doc2bow(text) for text in tokens_list]
	#print(corpus)

	#get model list and coherence values to plot the graph
	model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=csv_data['text_lemmatized'], start=20, limit=50, step=5)
	show_coherence_graph(coherence_values, 50, 20, 5)
	print('----EOF----')
	return True

