#pip install transformers

from summarizer import Summarizer


body = "This is a sample text that we want to summarize. It is a very long text with a lot of content. The goal is to generate a concise and informative summary of the text."

def summarize_bert(text):

    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")


    #body = 'Text body that you want to summarize with BERT'
    model = Summarizer()
    result = model(body, ratio=0.02) 
    print(result) # Specified with ratio
    result = model(body, num_sentences=1)  # Will return 3 sentences 
    print(result)

summarize_bert("")