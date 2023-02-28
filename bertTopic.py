from bertopic import BERTopic
from sklearn.datasets import fetch_20_newsgroups

# download dataset of 20,000 news articles
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# fit the topic model to the data, assigning a topic to each article
topic_model = BERTopic()
topic_model.fit_transform(docs)