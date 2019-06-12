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
from sklearn.metrics import precision_recall_fscore_support
'''df = pd.read_csv('raval_dataset_cleaned.csv')
for i in range(len(df['command'])):
	df['command'][i]=df['command'][i].split(" ")[0]
	
	
df.to_csv("base_class.csv", sep='\t',encoding='utf-8', index=False)'''

df = pd.read_csv('base_class.csv')
data_y=pd.read_csv("base_class.csv",usecols=["command"])
commands=data_y.command.tolist()
commands_set=list(set(commands))
d={}
for i in range(len(commands_set)):
	d[commands_set[i]]=i

X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['command'],test_size=0.1, random_state=0)
y_train=y_train.tolist()
y_test=y_test.tolist()
y_train_index=[]
for i in range(len(y_train)):
	y_train_index.append(d[y_train[i]])
y_test_index=[]
for i in range(len(y_test)):
	y_test_index.append(d[y_test[i]])

def cleanText(raw_text, remove_stopwords=True, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''
    #text = BeautifulSoup(raw_text, 'html.parser').get_text()  #remove html
    letters_only = re.sub("[^a-zA-Z.-]", " ", str(raw_text))  # remove non-character
    words = letters_only.lower().split() # convert to lower case

    if remove_stopwords: # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stemming==True: # stemming
#         stemmer = PorterStemmer()
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    if split_text==True:  # split text
        return (words)

    return( " ".join(words))
X_train_cleaned = []
X_test_cleaned = []


for j in X_train:
    X_train_cleaned.append(cleanText(j))

for j in X_test:
    X_test_cleaned.append(cleanText(j))

def modelEvaluation(predictions):
    '''
    Print model evaluation to predicted result
    '''
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_index, predictions)))
    #print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
    #print("\nClassification report : \n", metrics.classification_report(y_test_index, predictions))
    #print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test_index, predictions))
def modelTrainEvaluation(predictions):
    '''
    Print model evaluation to predicted result
    '''
    print ("\nAccuracy on train set: {:.4f}".format(accuracy_score(y_train_index, predictions)))
    #print("\nAUC score : {:.4f}".format(roc_auc_score(y_train, predictions)))
    #print("\nClassification report : \n", metrics.classification_report(y_train_index, predictions))
    #print("\nConfusion Matrix : \n",metrics.confusion_matrix(y_train_index, predictions) )

tfidf = TfidfVectorizer(min_df=1)
X_train_tfidf = tfidf.fit_transform(X_train_cleaned)
model=svm.SVC(gamma=0.5,C=0.1)
#model=svm.SVC(kernel="linear")

model.fit(X_train_tfidf,y_train_index)
predictions = model.predict(tfidf.transform(X_train_cleaned))
modelTrainEvaluation(predictions)
predictions = model.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)
print(precision_recall_fscore_support(y_test_index, predictions, average='micro'))
print(d)
#interactive part
a=1
while(a):
	a=int(input("enter 1 to continue or 0 to exit \n"))
	if(a==1):
		predictions=model.predict(tfidf.transform([input()]))
		for key, value in d.items() :
			if(value==predictions):
				print(key)
				break
	else:
		break


	   



