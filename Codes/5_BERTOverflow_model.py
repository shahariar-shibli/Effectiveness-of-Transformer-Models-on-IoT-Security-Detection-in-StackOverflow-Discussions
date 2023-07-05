from transformers import *
import numpy as np
import pandas as pd
import random as rd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report as cr 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from keras import Input,Model,callbacks

from bertclassifier_py import *
np.random.seed(10)
rd.seed(10)
tf.random.set_seed(10)
import scipy.sparse as sp

def random():
    from numpy.random import seed
    seed(1)
    import random
    random.seed(1)
    from tensorflow.random import set_seed
    set_seed(1)
    import os
    os.environ['PYTHONHASHSEED'] = '1'
    
iot = pd.read_excel('IoT_Security_Dataset.xlsx')
iot['Security'] = iot.Security.apply(lambda x: 0 if x=='0' else 1)
class BertModel:
    def __init__(self,label,aspect):
        self.num_class = label
#         self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#         self.model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
#         self.model = TFXLNetForSequenceClassificationTemp.from_pretrained('xlnet-base-cased')
#         self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.model = TFBertForSequenceClassificationTemp.from_pretrained('bert-base-uncased')
#         self.tokenizer  = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.tokenizer = BertTokenizer.from_pretrained('jeniya/BERTOverflow')
        self.model = TFBertForSequenceClassificationTemp.from_pretrained('../input/bertoverflow', from_pt = True)
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.model = TFRobertaForSequenceClassificationTemp.from_pretrained('roberta-base')
#         self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
#         self.model = TFDistilBertForSequenceClassificationTemp.from_pretrained('distilbert-base-cased')
#         self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
#         self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')

#         self.dataset = dataset
        self.current_aspect = aspect


    def re_initialize(self):
        
#         self.model = TFBertForSequenceClassificationTemp.from_pretrained('bert-base-uncased')
#         self.model = TFRobertaForSequenceClassificationTemp.from_pretrained('roberta-base')
#         self.model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
#         self.model = TFXLNetForSequenceClassificationTemp.from_pretrained('xlnet-base-cased')
#         self.model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
#         self.model = TFDistilBertForSequenceClassificationTemp.from_pretrained('distilbert-base-cased')
        self.model = TFBertForSequenceClassificationTemp.from_pretrained('../input/bertoverflow',  from_pt = True)
#         self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')


    
    
    def tokenize(self, dataset):
        input_ids = []
        attention_masks = []

        for sent in dataset:
            sent = str(sent)
            bert_inp = self.tokenizer .encode_plus(sent.lower(), add_special_tokens = True, max_length = 100, truncation = True, padding = 'max_length', return_attention_mask = True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])

        train_input_ids = np.asarray(input_ids)
        train_attention_masks = np.array(attention_masks)

        return [train_input_ids,train_attention_masks]
    
    def model_compilation(self, e):

        #print('\nAlBert Model', self.model.summary())


        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate = e, epsilon = 1e-08)
        self.model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    
    
    def run_model(self, batch=32, epoch=4, lr=1e-5):
        
        f1=[]
        recall = []
        precision = []
        auc = []
        mcc = []
        pred=[]
        actual = []
        prob = []
        for i in range(10):
                dataset = iot
                test_df, train_df = dataset[dataset['Run']==i], dataset[dataset['Run']!=i];

                train = self.tokenize(train_df['Sentence'].to_list())
                test = self.tokenize(test_df['Sentence'].to_list())
                train_y = train_df.Security.to_numpy()
                test_y = test_df.Security.to_numpy()



                self.re_initialize()
                self.model_compilation(lr)

                history = self.model.fit(train, train_y, batch_size = batch, epochs = epoch, validation_data = (test,test_y), shuffle=True)

                output = self.model.predict(test, batch_size = 1)
                
                temp = tf.keras.activations.sigmoid(output).numpy()[0]
                temp /= temp.sum(axis=-1)[:,np.newaxis]
                actual.extend(test_y)
                prob.extend(temp[:,1])



                final_output = np.argmax(output, axis = -1)[0]
    #             print(final_output, test_y)
                pred.extend(final_output)
                f1_score_val=f1_score(test_y, final_output, average = None)


                pre = precision_score(test_y, final_output, average = None)
                re = recall_score(test_y, final_output, average = None)
                ac = accuracy_score(test_y, final_output)
                rc=roc_auc_score(test_y, final_output)
                mc = matthews_corrcoef(test_y, final_output)

                f1.append(f1_score_val)
                precision.append(pre)
                recall.append(re)
                auc.append(rc)
                mcc.append(mc)
                del self.model
                del history

                print(f1_score_val)

        #print(f1)

        return f1,precision,recall,auc,mcc, pred, actual, prob
    def run_full_model(self, batch=32, epoch=4, lr=1e-5):
        
        f1=[]
        recall = []
        precision = []
        auc = []
        mcc = []
        pred=[]
        for i in range(1):
                train = self.tokenize(iot['Cleaned Sentence'].to_list())
                test = self.tokenize(validation.sentence.to_list())
                train_y = iot.Security.to_numpy()
                test_y = validation.IsAboutSecurity.to_numpy()


                self.re_initialize()
                self.model_compilation(lr)

                history = self.model.fit(train, train_y, batch_size = batch, epochs = epoch, validation_data = (test,test_y), shuffle=True)

                output = self.model.predict(test, batch_size = 1)




                final_output = np.argmax(output, axis = -1)[0]
    #             print(final_output, test_y)
                pred.extend(final_output)
                f1_score_val=f1_score(test_y, final_output, average = None)


                pre = precision_score(test_y, final_output, average = None)
                re = recall_score(test_y, final_output, average = None)
                ac = accuracy_score(test_y, final_output)
                rc=roc_auc_score(test_y, final_output)
                mc = matthews_corrcoef(test_y, final_output)

                f1.append(f1_score_val)
                precision.append(pre)
                recall.append(re)
                auc.append(rc)
                mcc.append(mc)
#                 del self.model
#                 del history

                print(f1_score_val)
        #print(f1)

        return f1,precision,recall,auc,mcc, pred
    
    def label_dataset(self):
        dataset = pd.read_csv('../input/iot-paper/StackOverflowIoTAllSentences.csv')
        temp = []
        for i in range(673):
            newdf = dataset[i*1000:min((i+1)*1000, 672678)]
            token = self.tokenize(newdf.Sentence.values)
            t = self.model.predict(token)
            t = np.argmax(t, axis = -1)[0]
            temp.extend(t)
            print(i)
        dataset['Security'] = temp
        dataset.to_csv('results.csv')
        d = dataset[dataset.Security==1]
        d.to_csv('IoTSecuritySentence.csv')
        return temp
            
            
        

#
class_count = 2
aspect = 10
bert = BertModel(class_count,aspect)
# bert.dataset = dataset_usa
#y=bert.tokenize(dataset['X_Test'][0])
lr = [1e-5]
batch = [32]
epochs = [2]
actual = []
prob = []
dictionary = {}
result=[]
for l in lr:
    for b in  batch:
        for e in epochs:
            random()
            x = 10
            f1,precision, recall, auc, mcc,result, actual, prob = bert.run_model(lr = l, batch = b, epoch = e)
            dictionary[(l,b,e)] = { 'pre': sum(precision)[1]/x,'re': sum(recall)[1]/x,'f1': sum(f1)[1]/x, 'auc': sum(auc)/x, 'mcc': sum(mcc)/x }
            print(dictionary[(l,b,e)])
