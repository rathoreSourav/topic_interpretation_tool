import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from random import sample
from typing import List
from summarizer import Summarizer
import pandas as pd
import streamlit as st

from gensim.summarization import summarize


#gensim summarize
from gensim.summarization import summarize

tokenizer = T5Tokenizer.from_pretrained("t5-base")
language_model = T5ForConditionalGeneration.from_pretrained("t5-base")
headline_generator = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")

def handler(text):
    #body = 'Text body that you want to summarize with BERT'
    model = Summarizer()
    summarized = " "
    try:
        summarized = summarize(text, word_count=400)

    except:
        summarized = " "         

    #bert summary enable when high gpu
    #summarized = model(text, num_sentences=3)

    #gensim summary, enable when low gpu
    #summarized = summarize(text)
    #print('summarized sentebnnce: ', summarized)
    return summarized  # Specified with ratio
    #result = model(body, num_sentences=3)  # Will return 3 sentences 


def generate_topic_label(summary_list, articles: List[str]) -> str:
    summarized_summary_list = []
    current_token_length = 0
    max_token_length = 512
    i = 0
    for sentence in summary_list:
        #st.write('summarizing article...', i)
        summarized_summary_list.append(handler(sentence))
        i += 1

    encoding = tokenizer.encode("headline: " + " ".join(summarized_summary_list), return_tensors="pt")
    output = headline_generator.generate(encoding)
    return tokenizer.decode(output[0][1:-1])


def generate_topic_label_udpated(summary_list):
    summarized_summary_list = []
    current_token_length = 0
    max_token_length = 512
    i = 0
    #st.write('total articles in this topic: ', len(summary_list))

    #progress_bar = st.progress(0)
    for sentence in summary_list:

        #st.write('summarizing article...', i)
        summarized_summary_list.append(handler(sentence))
        i = i + 1
        #progress_bar.progress(i)
       

    encoding = tokenizer.encode("headline: " + " ".join(summarized_summary_list), return_tensors="pt")
    output = headline_generator.generate(encoding)
    return tokenizer.decode(output[0][1:-1])    

def generate_label(topic_set, article_list):
    #topic_labels = {0:'technology',1:'company', 2:'digitalization',3:'business_model', 4:'resource', 5:'system', 6:'process', 7:'customer'}
    myset = {"technology", "company", "digitalization", "business_model", "resource", "system", "process", "customer"}
    return generate_topic_label_udpated(article_list)
    #print(f"Topic 13 label------------------------------------------------------------: {generate_topic_label(article_list, myset)}")

def main():
    datafile = pd.read_csv("extracted_doc.csv")
    filename = list(datafile['Text'])
    topic_sets=set(datafile['Keywords'].unique())
    print('topic_list: ', datafile['Keywords'].unique()[0])
    text=''
    count=0
    slist=[]
    myset = {"technology", "company", "digitalization", "business_model", "resource", "system", "process", "customer"}
   # print(f"Topic 13 label: {generate_topic_label(myset)}")
    for i in filename:
        count+= 1
        slist.append(i)
        text += i
        if(count == 10):
            text=''
            #count=0
            generate_label(datafile['Keywords'].unique()[0], slist)