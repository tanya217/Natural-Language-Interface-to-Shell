import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from models import InferSent
import torch
from scipy.stats.stats import pearsonr   
import numpy as np
df = pd.read_csv('base_class_flag1.csv',error_bad_lines=False)
#df = pd.read_csv('sampledata.csv',error_bad_lines=False)
#module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#embed = hub.Module(module_url)
#InferSent
V = 1
MODEL_PATH = 'infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = '/home/malathy/InferSent/dataset/GloVe/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)
index=0
corr=[]
count=0
flag=0
#inp=input("Enter NL Query \n")

for i in range(len(df['command'])):
	sentences=["Transfer file1.doc to fold1/file2.doc and overwrite if file2.doc exists."]
	sentences=[inp]
	
	if(re.search('^[mv*]',df['command'][i] )):
	#if("mv" in df['command'][i]):
	
		flag=1
		
		sentences.append(df['Description'][i])
		#print(sentences)
		
		infersent.build_vocab(sentences, tokenize=True)
		embeddings = infersent.encode(sentences, tokenize=True)
		#corr.append(np.inner(embeddings[0],embeddings[1]))
		corr.append(pearsonr(embeddings[0],embeddings[1])[0])
		
	else:
		if(flag==0):
			count=count+1
		else:
			break



maxx=corr[0]
for j in range(1,len(corr)):
	if(corr[j]>maxx):
		maxx=corr[j]
		index=j


print(df['command'][count+index])






#google's sentence encoder

'''corr=[]
index=0
for i in range(len(df['command'])):
    if("ifconfig" in df['command'][i]):
        sentence="Show all interactive network interface in a short list."
        sent=df['Description'][i]
        tf.logging.set_verbosity(tf.logging.ERROR)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(embed([sentence]))
            sent_emb=session.run(embed([sent]))
            #sentence = sentence.astype(np.float)
            corr.append(np.inner(sent_emb, message_embeddings))
        max=corr[0]
        for j in range(1,len(corr)):
            if(corr[j]>max):
                max=corr[j]
                index=j
        print(df['command'][i+index])'''
