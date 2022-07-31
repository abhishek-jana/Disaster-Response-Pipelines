import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['omw-1.4','punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as go
# from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

from wsgiref import simple_server

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    # Custom transformer to extract starting verb
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class StartingNounExtractor(BaseEstimator, TransformerMixin):
    # Custom transformer to extract starting Noun
    def starting_noun(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['NN', 'NNS', 'NNP', 'NNPS'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load("./models/classifier.pkl")

def return_figures():
    """Creates two plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the two plotly visualizations

    """
    
    graph_one = []
    category_names = df.iloc[:,4:].columns.values.tolist()
    value_1 = [df[df[category]==1][category].value_counts()[1] for category in category_names]
    value_0 = [df[df[category]==0][category].value_counts()[0] for category in category_names]
    graph_one.append(
        Bar(
        name = '1',
        x = category_names,
        y =  value_1,        
      )
    )
    graph_one.append(
        Bar(
        name = '0',
        x = category_names,
        y =  value_0,
       
      )
    )
    layout_one = dict(title = 'Distribution of Message Categories',
                xaxis = dict(title = "Category",),
                yaxis = dict(title = "Count"),
                )

    graph_two = [] 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_two.append(
      Bar(
        x = genre_names,
        y = genre_counts,
      )
    )

    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = "Genre",),
                yaxis = dict(title = "Count"),
                )


    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    
    
    return figures


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    figures = return_figures()
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

# def main():
#     app.run(host='0.0.0.0', port=port, debug=True)
#
#
# if __name__ == '__main__':
#     main()
