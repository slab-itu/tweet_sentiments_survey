import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

X = []
y = []
class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
        
def getTrainingAndTestData():
        

        #Training data 1: Sentiment 140
        f=open(r'/Users/aneelasaleem/Documents/ITU/Thesis/ThesisII/all_langs/data/altmetrics_data_cleaned.csv','r', encoding='ISO-8859-1')
        reader = csv.reader(f)
        next(reader, None) #skip header
        for row in reader:
            
            if row[3]=='positive' or row[3]=='neutral':
                y.append(1)
                X.append(row[2])
            elif row[3]=='negative':
                y.append(-1)
                X.append(row[2])
            
            
        print("X===length",len(X))
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.36, random_state=42)
        return X_train, X_test, y_train, y_test

# Linear SVM classifier
def SVMClassifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 1))
        svm_clf =svm.SVC(kernel='rbf', C=1000, gamma = 0.001, tol=2,class_weight='balanced')
        vec_clf = Pipeline([('TfidfVectorizer', vec), ('clf', svm_clf)])
        vec_clf.fit(X_train,y_train)
        return vec_clf

def predict(text):
    X_train, X_test, y_train, y_test = getTrainingAndTestData()
    vec_clf = SVMClassifier(X_train,y_train)
    target_names = {1:"postive", -1:"negative"}

    y_pred = vec_clf.predict(text)  
    for doc, category in zip(text, y_pred):
            print('%r => %s' % (doc, target_names[category]))
# Main function
def main(arg):
#        X_train, X_test, y_train, y_test = getTrainingAndTestData()
#        if arg=='svm':
#            vec_clf = SVMClassifier(X_train,y_train)
#        
#        # save the model to disk
#        filename = '/Users/aneelasaleem/Documents/ITU/Thesis/ThesisII/all_langs/code/Saved_Model/finalized_model.sav'
#        #pickle.dump(vec_clf, open(filename, 'wb'))
#
#        # load the model from disk
#        vec_clf = pickle.load(open(filename, 'rb'))
#        
        predict_me = ["you raise test scor even iq test practic test skill you lower test scor chang test", "world best acronym", "ild gudgeon gobio gobio french river contaminat microplastic preliminary study first evidence"];
      
        predict(predict_me)
#                    
        
if __name__ == "__main__":
   
    model=""
    main(arg=model)

    
