#pip install -U sentence-transformers

from summarizer import Summarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler

handler = CoreferenceHandler(greedyness=.4)
# How coreference works:
# >>>handler.process('''My sister has a dog. She loves him.''', min_length=2)
# ['My sister has a dog.', 'My sister loves a dog.']

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(sentence_handler=handler)
model(body)
model(body2)