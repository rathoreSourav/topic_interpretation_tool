import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

from gensim.summarization import summarize



def summarizess(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summarys=''.join(final_summary)
    print("summary of the document")
    print(summarys)
    return summarys


doc = " After the implementation of such a system, the consumers’ satisfaction with custom-made apparel bought online should be investigated based on several factors (e.g. personal characteristics,fashion involvement, body satisfaction, purchase behavior, general attitude towards novelties with an emphasis on new technologies).(ŽURAJ et al. 2017, p.10).the ecosystem may be defined as the part of the environment with which an organization interacts. Consequently, in a pragmatist view, the ecosystem is performed by the choices—deliberate, emergent or constrained—made by an organization concerning its business_model. firms have various choices for navigating nascent ecosystems. They may follow a positioning logic driven by the search for bargaining power, a competency logic driven by their pre-existing capabilities or a bottleneck logic driven by entering bottleneck components of the ecosystem to create value."

print(summarize(doc, ratio=0.5))