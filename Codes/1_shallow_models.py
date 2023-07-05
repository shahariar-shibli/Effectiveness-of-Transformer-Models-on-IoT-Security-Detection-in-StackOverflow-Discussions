import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

iot = pd.read_excel('IoT_Security_Dataset.xlsx')

def VocabularyCleaner(vocabulary):
        number=r"^[0-9]+"
        number=re.compile(number)
        specialCharacter=r'^[@_!#$%^&*()<>?/\|}{~:]+$'
        specialCharacter=re.compile(specialCharacter)
        urls=r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        urls=re.compile(urls)
        l=list(filter(urls.match,vocabulary))
        l.extend(list(filter(number.match,vocabulary)))
        l.extend(list(filter(specialCharacter.match,vocabulary)))
        return l

countvector = CountVectorizer(stop_words='english')
def classifier(type):
  if type=="SGD":
    cls = SGDClassifier(alpha=.0001, max_iter=2000,
                                          epsilon=0.5, loss='log',penalty="l2",
                                          power_t=0.5, warm_start=False, shuffle=True, class_weight='balanced')
    return cls
  cls = LogisticRegression(max_iter=2000,penalty='l2',warm_start=False,class_weight='balanced')
  return cls

def Sentence2VectPreparation(sentTrain,sentTest):
        tfidfTransformer = TfidfTransformer()
#         print(sentTrain, sentTest)
        
        tfidVect=TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english')
        tfidVect.fit(sentTrain)
        print(tfidVect.vocabulary_.__len__())
        xTrain = tfidVect.transform(sentTrain)

        xTrain = tfidfTransformer.fit_transform(xTrain)

        xTest = tfidVect.transform(sentTest)
        xTest = tfidfTransformer.fit_transform(xTest)

        return xTrain,xTest

pre =[]
re=[]
F1=[]
auc = []
mccc = []
def PerformanceMatrix(yTest,prediction):
        TN, FP, FN, TP = confusion_matrix(yTest, prediction).ravel()
        total=len(yTest)
        acc=(TP+TN)/total
        sn=TP/(TP+FP)#precision
        sp=TP/(TP+FN)#recall
        mcc=(TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
        f1=2*(sn*sp)/(sn+sp)
        neg=TN/(TN+FP)
        ac = roc_auc_score(yTest, prediction)
        mc = matthews_corrcoef(yTest, prediction)
        mccc.append(mc)
        auc.append(ac)
        pre.append(sn)
        re.append(sp)
        F1.append(f1)
        print(sn,sp,f1, ac, mcc)

def cross_fold_performance(dataset, label):
  dictionary_k_folds = cross_fold(dataset, label)
  pre =[]
  re=[]
  F1=[]
  auc = []
  mccc = []
  for k in range(10):
    current_k = k
    x_train=iot[iot.Run!=current_k].Sentence
    x_test=iot[iot.Run==current_k].Sentence
    y_train=iot[iot.Run!=current_k].Security
    y_test=iot[iot.Run==current_k].Security

    xTrain,xTest=Sentence2VectPreparation(x_train,x_test)
    #select the classifier
    cls = classifier("SGD")
    
    cls.fit(xTrain, y_train)
    prediction = cls.predict(xTest)
    PerformanceMatrix(y_test,prediction)

  print(sum(pre)/10, sum(re)/10,sum(F1)/10 , sum(auc)/10, sum(mccc)/10)
