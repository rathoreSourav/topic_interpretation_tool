__author__      = "Sourav Singh"
__version__ = "1.0.1"
__project__ = "this a home page for Topic Interpretaion tool"

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#importing wordnet lemmatizer for linking words with similar meanings to one word
from nltk.stem.wordnet import WordNetLemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
import model_topic
import string
from PIL import Image
from remove_stopwords import run_preprocess
from insights_vis import insights_visulaize

from rem_non_english import clean_column
from inputUrlandName import takeUserData
from scraper import do_scrap
import word_dist 


# multiwords as list
multiwords_list=[]
not_cleaned = True

#lemmatization function
def lemmatize_data(csv_data):
	st.write("lemmatizing the corpus data...")
	lemmatizer = nltk.stem.WordNetLemmatizer()
	lem_word = [lemma.lemmatize(word) for word in lem_word]
	return lem_word

#remove punctions from data
#creating a new column in csv as content_processed for storing removed punctuation, lowered case and stopwords free data
def punc_remover(csv_data):
	st.write('cleaning data for any punctuation...')
	csv_data['content'] = csv_data['content'].fillna(value='missing_data')
	#punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
	st.write('removing non english data from content...')
	csv_data['content'] = clean_column(csv_data['content'])
	#st.write(csv_data.head())
	csv_data['content_processed'] = \
	csv_data['content'].map(lambda x: str(x).translate(str.maketrans('', '', string.punctuation)))
	#remove any numbers
	csv_data['content_processed'] = csv_data['content_processed'].str.replace('\d+', '')
	#csv_data['content'].map(lambda x: re.sub('[,\.!?]', '', x))
	csv_data = csv_data.replace('”', '')
	csv_data = csv_data.replace('“', '')
	csv_data = csv_data.replace('(', '')
	csv_data = csv_data.replace(')', '')
	#print(csv_data)
	return csv_data


# define a function to lemmatize the text
def lemmatize_text_data(text):
	lemmatizer = WordNetLemmatizer()
	words = nltk.word_tokenize(text.lower())  # tokenize the text into words
	lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # lemmatize each word
	return " ".join(lemmatized_words)  # join the lemmatized words back into a string


#data transformation to lower
def to_lower(csv_data):
	st.write("converting corpus content to lower...")
	csv_data['content_processed_lower'] = \
	csv_data['content_processed'].map(lambda x: x.lower())

	# apply the lemmatize_text function to the 'text' column and create a new column 'lemmatized_text'
	csv_data['content_processed_lower'] = csv_data['content_processed_lower'].apply(lemmatize_text_data)

	return csv_data

#multiwords processing
def multiwords_processor(csv_data, multiwords_texts):
	st.write('processing multiwords...')
	temp_text_list = []
	multiwords_texts_dict ={}
	for multiwords in multiwords_texts:
		#print(multiwords)
		multiwords_texts_dict[multiwords.replace(" ", "_")] = multiwords

	print(multiwords_texts_dict)
	for i, j in multiwords_texts_dict.items():
		#print(i, j)
		csv_data = csv_data.replace(j, i, regex=True)
		#print(csv_data)

	#print(csv_data)
	return csv_data

#exploratory analysis to identify the stopwords by looking at wordcloud
def wordcloud_generator(csv_data_column):
	st.write("generating wordcloud....")
	#create all sententenc in a string
	contents_to_string = ','.join(list(csv_data_column.values))
	#Create a WordCloud object
	# Generate a word cloud image
	wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False).generate(contents_to_string)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig('wordcloud.png')
	image = Image.open('wordcloud.png')
	st.image(image, caption='Wordcloud')


#loading english dictionary for stopwordsextends
stop_words = set(stopwords.words('english'))

#remove stopwords
def remove_stopwords(csv_data):
	#try:
	stop_words_session = st.session_state['stopwords']
	print("stopwrds are: ", stop_words_session)
	st.write("removing stopwords from corpus data...")
	csv_data['content_processed_stops'] = \
	csv_data['content_processed_lower'].apply(lambda x: ' '.join([word for word in x.split() if word.lower().strip() not in (stop_words_session)]))
	#removes any words less than three characters after stopwords removal 
	csv_data['content_processed_stops'].map(lambda x: re.sub(r'\b\w{1,4}\b', '', x))
	csv_data.to_csv('cleaned_doc.csv', index=False)
	run_preprocess()
	csv_data = pd.read_csv('output.csv')
	#except:
	#	print("except block of remove stopwords")	
	return csv_data


	
def lemmatize_text(text):
	lemmatizer = nltk.stem.WordNetLemmatizer()
	return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#main data cleaner
def data_cleaning_processor(raw_data):
	cleaned_data = punc_remover(raw_data)
	cleaned_data = to_lower(cleaned_data)
	#wordcloud_generator(cleaned_data)
	cleaned_data = multiwords_processor(cleaned_data, multiwords_list)
	cleaned_data = remove_stopwords(cleaned_data)
	#cleaned_data = lowered_data.drop('content_processed', 1)
	st.write("lemmatizing text...")
	cleaned_data['text_lemmatized'] = cleaned_data.content_processed_stop.apply(lemmatize_text)
	cleaned_data.to_csv('cleaned_doc.csv', index=False)
	#print(cleaned_data)
	st.write(cleaned_data.head())
	not_cleaned = False
	return cleaned_data

###end of data cleaning code####


# Set page config
st.set_page_config(page_title="Topic Interpretaion Tool", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")

# Change background color
st.markdown(
    """
    <style>
    body {
        background-color: #e31010;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Topic Interpretaion Tool")
st.sidebar.title("Quick Navigation")

quick_navigation = st.sidebar.selectbox('Select to navigate',
									("Home", "Wordcloud", "Model Score", "Visualize Result", "Insights", "Web Scrapper", "About Us"))

if quick_navigation == 'Home':
	data = pd.DataFrame()
	stopword_file = ''
	multiwords_file = []


	if data.empty:
		st.subheader("please upload the companies csv data to continue...")

		data_file = st.file_uploader("",type=["csv"])
		if data_file is not None:
			file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}
			st.write(file_details)
			data = pd.read_csv(data_file)
			st.dataframe(data.head())
			st.session_state['data'] = data
			


	if not data.empty:
		st.subheader("please upload stopword txt file*")
		docx_file = st.file_uploader("Upload Stopwords", type=["txt"])
		if docx_file is not None:

			file_details = {"filename":docx_file.name, "filetype":docx_file.type,
                            "filesize":docx_file.size}
			st.write(file_details)
			#raw_text = str(docx_file.readlines(),"utf-8")
			stopword_file = set(line.decode('utf-8').rstrip('\n') for line in docx_file or [])
			#stopword_file = raw_text
			stop_words.update(stopword_file)
			#st.text("processed stopwords...")
			st.session_state['stopwords'] = stop_words
			

	if not data.empty:
		st.subheader("please upload multiwords txt file*")
		docx_file1 = st.file_uploader("Upload Multiwords", type=["txt"])
		#if st.button("Process Multiwords"):
		if docx_file1 is not None:
			file_details1 = {"filename":docx_file1.name, "filetype":docx_file1.type,
                            "filesize":docx_file1.size}
			st.write(file_details1)
			multiwords_list = set(line.decode('utf-8').rstrip('\n') for line in docx_file1 or [])
			#st.text("processed multiwords...")
			st.session_state['multiwords'] = multiwords_list


	if st.button('Clean Corpus'):		
		try:	
			if(len(st.session_state['multiwords'])>0 and len(st.session_state['stopwords'])>0 and len(st.session_state['data'])>0):
				with st.spinner("Cleaning corpus..."):
					cleaned_data = data_cleaning_processor(data)
					#storing data to local csv for further verification 
					cleaned_data.to_csv('./output/cleaned_data.csv', header=True, index=False)
					st.session_state['cleaned_data'] = cleaned_data
					#st.write(st.session_state['cleaned_data'].head())
					st.subheader("Data cleaned!")
					st.write('A copy of cleaned csv can be found at: ./output/cleanded_data.csv')
					st.write()
					st.session_state['data_cleaned'] = True
			

		except:
		 	#st.write(ValueError)
		 	st.write("Please upload the required data first")

	try:
		if st.session_state['data_cleaned']:	
			st.markdown("---")
			st.subheader("Topic Model")
			value = st.text_input("Enter number of topics values 1-50") 
			#st.slider("number of topics", min_value=0, max_value=50, value=0, step=1)
			st.session_state['value'] = value
	except:
		print('data yet to be cleaned')		
					
	try:
		if(st.session_state['value']):
			print("proceeding with topic modeling and number of topic selected as:", st.session_state['value'])
			if int(st.session_state['value']) > 0:
				st.write('You have selected ', value, ' topics, please click below to proceed')
				if st.button('PROCEED'):
					st.write('procedding...')
					with st.spinner("Topic modelling"):
						cleaned_data = st.session_state['cleaned_data']
						#st.write('proceeding with topic modelling.....')
						model_topic.topic_modelling(cleaned_data, int(value))

			else:
				print("else block...do nothing...")
	except:
		print("Upload to continue!")
		#st.error(ValueError)

	try:
		if(st.session_state['lda_model']):
			st.subheader("Topic documents")
			#doc_cut_off = st.text_input("Enter max number of asscoiated documents ny default it is 25") 
			#st.slider("number of topics", min_value=0, max_value=50, value=0, step=1)
			#st.session_state['doc_cut_off'] = doc_cut_off
			if st.button('GET DOCUMENTS'):
				csv_data = st.session_state['cleaned_data']
				corpus = st.session_state['corpus']
				lda_model = st.session_state['lda_model']
				with st.spinner("Fetching associated documents"):
					model_topic.get_doc_for_topics(csv_data = csv_data, corpus = corpus, ldamodel = lda_model, doc_cut_off=.3)
					st.session_state['is_document_available'] = True	

	except:
		print("EO except---do nothing")
		#st.write("Please upload the data!")	
			
	try:
		if st.session_state['is_document_available']:
			st.subheader("please provide topic percentage cut off for your your document!")
			doc_cut_off = st.text_input("Enter cut off, example 0.85, please save a copy of previous document, it will overwrite the existing doc") 
			st.session_state['doc_cut_off'] = doc_cut_off
			st.write("selected cut off: ", doc_cut_off)
			if st.session_state['doc_cut_off']:

				if st.button('PROCEED DOC CUT OFF'):
					with st.spinner("Cutting off documents"):
						try:
							float(doc_cut_off)
						except:
							st.write("Only float values are alllowed for cut off, incorrect value: ", doc_cut_off)	
						csv_data = st.session_state['cleaned_data']
						corpus = st.session_state['corpus']
						lda_model = st.session_state['lda_model']	
						model_topic.get_doc_for_topics(csv_data = csv_data, corpus = corpus, ldamodel = lda_model, doc_cut_off=float(doc_cut_off))
						st.success("success!")

				
			if st.session_state['doc_cut_off']:
				st.markdown("---")
				with st.spinner("Generating Topic Labels!"):
					if st.button('Generate Topic Labels'):		
						model_topic.auto_labelling_generation()
						st.markdown("---")



			if st.session_state['doc_cut_off']:
				if st.button('Generate Topic Summary'):		
					model_topic.summary_extract_updated()
	except:
		print("EO except---do nothing")
		#st.write("Please upload the data!")				
			


if quick_navigation == 'Data Visualization':
	st.header("Data Visualization")
	try:
		st.subheader("Original corpus:")
		st.dataframe(st.session_state['data'])

	except:
		st.write("Data not uploaded yet!")

	try:
		if not st.session_state['cleaned_data'].empty:
			st.subheader("Cleaned Corpus")
			st.dataframe(st.session_state['cleaned_data'])	

	except:
		st.write("Corpus yet to be cleaned")

if quick_navigation == 'Model Score':
	st.subheader("LDA Model Coherence Score")
	#try:
	with st.spinner("computing model score..."):
		if not (st.session_state['cleaned_data'].empty):
			model_list, coherence_values = model_topic.get_coherence_score(st.session_state['cleaned_data'])
			model_topic.show_coherence_graph(coherence_values=coherence_values, start=5, limit=50, step=5)

		else:
			st.write("data needs to be cleaned!")

	#except:
	#	st.write("something went wrong, please run the data cleaner first from home page!")

if quick_navigation == 'Visualize Result':
	st.subheader("Topic Model Result Visualization")
	model_topic.visulize_result()


if quick_navigation == 'Insights':
	st.subheader("Topic Model Insights")
	insights_visulaize()
	st.markdown("---")
	word_dist.word_count_dist_whole()
	st.markdown("---")
	word_dist.word_count_dist_by_topic()

if quick_navigation == 'Wordcloud':
	#try:
	word = pd.read_csv('./output/cleaned_data.csv')
	wordcloud_generator(word['content_processed_stop'])
	st.success('Wordcloud generated')
	#except:
	#	st.write('Please clean the data first...')	


if quick_navigation == 'Web Scrapper':
	st.subheader("Web Scrapper for Topic Interpretaion")
	takeUserData()

	try:
		if st.session_state['listsaved'] == True:
			if st.button('Start Scraping'):
				with st.spinner("Scraping data..., this may take a while!"):
					with st.expander("Click to see scraping progress..."):
						do_scrap()
				st.success("Data Scrapped Successfully and CSV saved at ./input/scraped_data.csv")

	except:
		print('Companies not saved yet')


