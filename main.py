from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
import csv
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import warnings

df = pd.read_excel('Mental_health_dataset.xlsx')
print('====Depression Data=========')
print(df)
print('=====Depression Type=========')
df["label"].value_counts()
df = df.dropna()
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import string
nltk.download('wordnet')
def text_transformation(text):
    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'[^\w\s]', '', text)  # removes punctuation safely

    text = text.strip()

    return text
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}
def text_cleaner(text):
    if isinstance(text, str):  # Check if text is a string
        newString = text.lower()
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"', '', newString)
        newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = re.sub('[m]{2,}', 'mm', newString)
        return newString
    else:
        return ''
df['cleaned'] = df["text"].apply(text_cleaner)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
# Splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(df.cleaned, df.label, test_size=0.3)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.naive_bayes import MultinomialNB
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_dtm,y_train)
svm_score=svm.score(X_test_dtm,y_test)
y_predict=svm.predict(X_test_dtm)
print(y_predict)
y_true=y_test
print('Accuracy of SVM: '+ str(svm_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of SVM: '+(str(precision)))
print('Recall of SVM: '+(str(recall)))
print('F1-score of SVM: '+(str(fscore)))
print(classification_report(y_true,y_predict))
##cm=confusion_matrix(y_true,y_predict)
##f,ax=plt.subplots(figsize=(5,5))
##sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
##plt.xlabel("y_pred")
##plt.ylabel("y_true")
##plt.show()
import requests
your_api_key = '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'

def query_openai_chat(prompt):
    url = "https://api.openai.com/v1/chat/completions"  # Updated to use the chat completions endpoint
    headers = {
        "Authorization": f"Bearer {your_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",  # Example: Using GPT-3.5-turbo or specify another chat model available
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/chat_depression',methods=['POST','GET'])
def chat_depression():
    second_prediction=''
    _description=''
    FData1=''
    FData2=''
    FData3=''
    FData4=''
    fpredict=''
    if request.method=='POST':
        inputdata = request.form['text']
        FData1=inputdata
        
        updated_data = {
                'text': str(inputdata)
                }
        df = pd.DataFrame.from_dict(updated_data,orient='index')
        df = df.transpose()
        df['Clean_Text'] = df["text"].apply(text_cleaner)
        fpre=np.array(df['Clean_Text'])
        FData2=fpre[0]
        input_data_features = vect.transform(df['Clean_Text'])
        prediction = svm.predict(input_data_features)
        prediction=prediction[0]
        if prediction==0.0:
               fpredict="No Depression"
        else:
               fpredict="Depression"
               response = query_openai_chat("Solving for Depression:"+FData2)
               FData4=response['choices'][0]['message']['content']
               
    return render_template('chat_depression.html',second_prediction=fpredict,FData1=FData1,FData2=FData2,FData3=FData3,FData4=FData4)
#################################>>>>>>>>>>>>>>>>>>USER<<<<<<<<<<<<<<<<############################################################

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
