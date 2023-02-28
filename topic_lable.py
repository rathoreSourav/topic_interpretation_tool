from gensim import corpora, models

def auto():
	# Preprocess text data
	text_data = ["some text data to classify", "some more text data to classify"]
	#text_data_preprocessed = preprocess_text(text_data)

	# Create dictionary and corpus
	dictionary = corpora.Dictionary(text_data)
	corpus = [dictionary.doc2bow(text) for text in text_data]

	# Train LDA model
	num_topics = 5
	lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

	# Predict topic for new text
	new_text = read('textData.txt')
	#new_text_preprocessed = preprocess_text([new_text])
	new_text_bow = dictionary.doc2bow(new_text)
	topic = lda_model[new_text_bow]

	#label topic
	#technology, company, digitalization, digital, business_model, resource, business_models, system, process, customer
	topic_labels = {0:'technology',1:'company', 2:'digitalization',3:'business_model', 4:'resource', 5:'system', 6:'process', 7:'customer'}
	print(topic_labels[topic[0][0]])

def read(fileName):
	with open(fileName, 'r') as file:
	    text = file.read()
	    return text;

auto()
