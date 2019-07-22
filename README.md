# senti_altmetrics
A basic flask application for sentiment analysis of Altmetrics dataset. An SVM based model is build using sklearn library of python. TfidfVectorizer has been used with BOW features. 

# Installation
    * pip install flask (see how to create how to create Flask app https://flask.palletsprojects.com/en/1.1.x/installation/)
    * export FLASK_APP=sentialtmetrics.py
    * flask run
    
   Running on http://127.0.0.1:5000/ 

# Usage
access the API endpoint http://127.0.0.1:5000/classifyaltmetrics

params: data (JSON array)

# Sample Input
      
    {"data": 
    ["*sigh* When a science acronym is too distracting to even read the science! LOL http://t.co/p2OFAgnuwv @InvaderXan",
     "you raise test scor even iq test practic test skill you lower test scor chang test", 
     "world best acronym", 
     "ild gudgeon gobio gobio french river contaminat microplastic preliminary study first evidence"]}
# Sample Output
 
     [{"text": "*sigh* When a science acronym is too distracting to even read the science! LOL http://t.co/p2OFAgnuwv @InvaderXan", "sentiment": "negative"}, 
     {"text": "you raise test scor even iq test practic test skill you lower test scor chang test", "sentiment": "postive"},
     {"text": "world best acronym", "sentiment": "negative"}, 
     {"text": "ild gudgeon gobio gobio french river contaminat microplastic preliminary study first evidence", "sentiment": "postive"}]
             

