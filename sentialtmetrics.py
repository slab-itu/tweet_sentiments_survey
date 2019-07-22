import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os.path
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import digits
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
CORS(app)

class AltmetricsClassifier(object):
    X = []
    y = []
    vec_clf = None
    filename = 'svm_model.sav'

    def clean(data):
        cachedStopWords = stopwords.words("english")
        
        cleanedData = []
        for text in data:
            # remove stop words
            text = ' '.join([word for word in text.split() if word not in cachedStopWords])
            #remove urls
            text = re.sub(r'[a-z]*[:.]+\S+', '', text)
            #remove special characters and numbers
            remove_digits = str.maketrans('', '', digits)
            text=text.translate(remove_digits)
            text=text.replace('RT|:|#|;|\.|\||\[|\]|\(|\)|@|\\|\/',' ').replace('&|&','and').replace('>','').replace('\n',' ').strip()
            text=re.sub(r'\W+', ' ', text)
            cleanedData.append(text)
        return cleanedData

    def getTrainingAndTestData(self):
            f=open(r'/Users/aneelasaleem/Documents/ITU/Thesis/ThesisII/all_langs/data/altmetrics_data_cleaned.csv','r', encoding='ISO-8859-1')
            reader = csv.reader(f)
            next(reader, None) #skip header
            for row in reader:                
                if row[3]=='positive' or row[3]=='neutral':
                    self.y.append(1)
                    self.X.append(row[2])
                elif row[3]=='negative':
                    self.y.append(-1)
                    self.X.append(row[2])
                
                
            X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.36, random_state=42)
            return X_train, X_test, y_train, y_test

    # Linear SVM classifier
    def trainSVMClassifier(self, X_train,y_train):
            vec = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 1))
            svm_clf =svm.SVC(kernel='rbf', C=1000, gamma = 0.001, tol=2,class_weight='balanced')
            vec_clf = Pipeline([('TfidfVectorizer', vec), ('clf', svm_clf)])
            vec_clf.fit(X_train,y_train)
            
            # save the model to disk
            pickle.dump(vec_clf, open(self.filename, 'wb'))
            
            return vec_clf

    def classify(self, data):
        print(data)
        text = self.clean(data) #clean the data
        print(text)
        if os.path.isfile(self.filename):
            # load the model from disk
            vec_clf = pickle.load(open(self.filename, 'rb'))
        
        else:
            X_train, X_test, y_train, y_test = self.getTrainingAndTestData(AltmetricsClassifier)
            vec_clf = self.trainSVMClassifier(AltmetricsClassifier, X_train,y_train)
        
        target_names = {1:"postive", -1:"negative"}
        
        y_pred = vec_clf.predict(text)  
        
        arr = []
        for doc, category in zip(data, y_pred):
            arr.append({'text': doc,'sentiment':target_names[category]})
        
        return arr

@app.route("/classifyaltmetrics", methods=['POST'])
def classify_altmetrics():
     predict_me = request.json['data']
     res = AltmetricsClassifier.classify(AltmetricsClassifier, predict_me)
     return json.dumps(res)
