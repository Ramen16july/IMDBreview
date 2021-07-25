from flask import Flask,render_template,url_for,request
import requests
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import pickle

import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import nltk
nltk.download('stopwords')

#imdb_data=pd.read_csv ("IMDB Dataset.csv")

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
#imdb_data['review']=imdb_data['review'].apply(denoise_text)

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
#imdb_data['review']=imdb_data['review'].apply(remove_special_characters)

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
#imdb_data['review']=imdb_data['review'].apply(simple_stemmer)

#set stopwords to english
#stop=set(stopwords.words('english'))

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
#imdb_data['review']=imdb_data['review'].apply(remove_stopwords)

#norm_train_reviews=imdb_data.review[:40000]
#norm_test_reviews=imdb_data.review[40000:]

#cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
#cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
#cv_test_reviews=cv.transform(norm_test_reviews)

#Tfidf vectorizer
#tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
#tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
#tv_test_reviews=tv.transform(norm_test_reviews)

#labeling the sentient data
#lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(imdb_data['sentiment'])
#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(train_sentiments)

#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)

#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)

#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)

#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)



with open('Countvector.pickle','wb') as f:
    pickle.dump(cv,f)

with open('logistic.pickle','wb') as f:
    pickle.dump(lr,f)
    
with open('Countvector.pickle','rb') as f:
    cv=pickle.load(f)
    
with open('logistic.pickle','rb') as f:
    clf=pickle.load(f)
## 1 positive 0 negative
sample=["""A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only "has got all the polari" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams\' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master\'s of comedy and his life. <br /><br />The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional \'dream\' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell\'s murals decorating every surface) are terribly well done."""]

sample=[""""Hickory Dickory Dock was a good Poirot mystery. I confess I have not read the book, despite being an avid Agatha Christie fan. The adaptation isn't without its problems, there were times when the humour, and there were valiant attempts to get it right, was a little overdone, and the events leading up to the final solution were rather rushed. I also thought there were some slow moments so some of the mystery felt padded. However, I loved how Hickory Dickory Dock was filmed, it had a very similar visual style to the brilliant ABC Murders, and it really set the atmosphere, what with the dark camera work and dark lighting. The darker moments were somewhat creepy, this was helped by one of the most haunting music scores in a Poirot adaptation, maybe not as disturbing as the one in One Two Buckle My Shoe, which gave me nightmares. The plot is complex, with all the essential ingredients, though not as convoluted as Buckle My Shoe,and in some way that is a good thing. The acting was very good, David Suchet is impeccable(I know I can't use this word forever but I can't think of a better word to describe his performance in the series) as Poirot, and Phillip Jackson and Pauline Moran do justice to their integral characters brilliantly. And the students had great personalities and well developed on the whole, particularly Damian Lewis as Leonard. All in all, solid mystery but doesn't rank along the best. 7.5/10 Bethany Cox """]

sample=["""Awesome"""]
type(sample)
sample=str(sample)
sample=strip_html(sample)
sample=remove_special_characters(sample)
sample=simple_stemmer(sample)
sample=remove_stopwords(sample)
#type(sample)
sample=[sample]

sample=cv.transform(sample)
predict=clf.predict(sample)

int(predict)

print(clf.predict(sample))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        review = request.form["message"""]
        sample=str(review)
        sample=strip_html(sample)
        sample=remove_special_characters(sample)
        sample=simple_stemmer(sample)
        sample=remove_stopwords(sample)
        sample=[sample]
        vect = cv.transform(sample)
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)




