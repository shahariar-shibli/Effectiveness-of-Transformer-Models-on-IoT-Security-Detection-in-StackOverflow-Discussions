import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import nltk

import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel

from gensim.models import CoherenceModel
import spacy
from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess

nltk.download('stopwords')

def process_words(texts, stop_words,nlp,bigram_mod, trigram_mod, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in
             texts]

    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    texts_out = []

    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])

    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc
                 in texts_out]

    return texts_out
    
df = pandas.read_csv('IoT_Security_dataset.csv')

data = list(df.sentence)

bigram = gensim.models.Phrases(data, min_count=20, threshold=100)
trigram = gensim.models.Phrases(bigram[data], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = nltk.corpus.stopwords.words('english')

data_ready = process_words(data,stop_words,nlp,bigram_mod,trigram_mod)
id2word = corpora.Dictionary(data_ready)
print('Total Vocabulary Size:', len(id2word))


corpus = [id2word.doc2bow(text) for text in data_ready]
dict_corpus = {}

for i in range(len(corpus)):
    for idx, freq in corpus[i]:
        if id2word[idx] in dict_corpus:
            dict_corpus[id2word[idx]] += freq
        else:
            dict_corpus[id2word[idx]] = freq

dict_df = pandas.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])


dict_df.sort_values('freq', ascending = False).head(10)
extension = dict_df[dict_df.freq > 1000].index.tolist()

ids = [id2word.token2id[extension[i]] for i in range(len(extension))]
id2word.filter_tokens(bad_ids=ids)


stop_words.extend(extension)
# rerun the process_words function
data_ready = process_words(data,stop_words,nlp,bigram_mod,trigram_mod)
# recreate Dictionary
id2word = corpora.Dictionary(data_ready)
print('Total Vocabulary Size:', len(id2word))

id2word.filter_extremes(no_below=10, no_above=0.5)
print('Total Vocabulary Size:', len(id2word))

corpus = [id2word.doc2bow(text) for text in data_ready]


mallet_path = 'mallet-2.0.8/bin/mallet'

def coherence_values_computation(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(
             mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(
              model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
    
model_list, coherence_values = coherence_values_computation (
   dictionary=id2word, corpus=corpus, texts=data_ready, 
   start=1, limit=50, step=4)
limit=50; start=1; step=4;


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " is having Coherence Value of", round(cv, 4))
    
    
ldamallet = gensim.models.wrappers.LdaMallet(
             mallet_path, corpus=corpus, num_topics=9, id2word=id2word)

coherencemodel = CoherenceModel(
              model=ldamallet, texts=data_ready, dictionary=id2word, coherence='c_v')

ldamallet = model_list[2]

pickle.dump(ldamallet, open("ldamallet.p", "wb"))

tm_results = ldamallet[corpus]

df_weights = pandas.DataFrame.from_records([{v: k for v, k in row} for row in tm_results])
df_weights.columns = ['Topic ' + str(i) for i in range(1,10)]
topic = list(df_weights.idxmax(axis=1))

df['Topics'] = topic
score = list(df_weights.max(axis=1))
df['Correlation Score'] = score

df.to_excel('IoT_security_topics.xlsx')
