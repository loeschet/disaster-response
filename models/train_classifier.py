from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from os.path import abspath
import sys
import pickle
import time


# Custom Transformer to select "message" column
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - absolute filepath of SQL database file

    OUTPUT:
    X - a pandas dataframe containing the tweet messages and the
        corresponding genres
    Y - a pandas dataframe containing the labels for each of the disaster types
    Y.columns - the column names (i.e. the descriptions of disaster type)
    
    Given an absolute filepath to a SQL database file, this function created a
    pandas dataframe from the provided database and extracts the "message" and
    "genre" columns into an input feature dataframe and extracts the one-hot
    encoded labels for the different disaster categories into a separate label
    dataframe.
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df[["message", "genre"]]
    Y = df.loc[:, (df.dtypes == int) & (df.columns != "id")]

    return X, Y, Y.columns


def tokenize(text):
    '''
    INPUT:
    text - input text to be tokenized

    OUTPUT:
    lemmed_tokens - list of lemmatized tokens created from the input text 
    
    This function takes an input text string and first tokenizes and then
    lemmatizes it using common NLTK functions. Additionally, URLs are
    identified and replaced by the string "urlplaceholder".
    '''
    
    # in case the tweets contain links, find and replace these URLs with
    # placeholder
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
                 '|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # normalize to lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize
    tokens = word_tokenize(text)

    # lemmatize and keep only non stop words
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

    # lemmatize verbs
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens]

    return lemmed_tokens


def build_model():
    '''
    OUTPUT:
    pipeline - complete scikit-learn pipeline model, including column selector,
    word tokenizer and multi-output classifier
    
    This function builds the scikit-learn pipeline used for tweet disaster
    classification and returns it.
    '''

    pipeline = Pipeline([
    ('selector',  ColumnSelector(columns="message")),
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('topics_pipeline', Pipeline([
            ('counts', CountVectorizer(tokenizer=tokenize)),
            ('lda', LatentDirichletAllocation(n_components=10))
        ]))
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - scikit-learn model to be evaluated
    X_test - input features of test set
    Y_test - labels of test set
    category_names - names of classes that examples are classified into
    
    This function takes a scikit-learn model, features and labels of the test
    set as well as the disaster category names and prints out a classification
    report for the model, showing different perfomance metrics for each of the
    classes as well as different aggregations thereof.
    '''

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    INPUT:
    model - scikit-learn model to be saved
    model_filepath - path where to store the model
    
    This simple function stores a scikit-learn model in a pickle file.
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    if len(sys.argv) == 3:
        # load data from SQL database
        database_filepath, model_filepath = sys.argv[1:]
        database_filepath = abspath(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        # build model
        print('Building model...')
        model = build_model()

        # simple hyperparameter tuning
        print('Find best parameters using simple grid search')
        print('NOTE! This can take ~1 1/2 hrs. on a modern-day CPU!')
        parameters = {"clf__estimator__max_depth": [3, 20, None],
                      "clf__estimator__max_features": ['sqrt', 'log2'],
                      "clf__estimator__n_estimators": [10, 20, 30]}

        cv = GridSearchCV(model, parameters, scoring="f1_macro", refit=False,
                          verbose=3)

        start = time.time()
        cv.fit(X, Y)
        end = time.time()
        
        print(f"Grid search took {(end - start)/60:.2f} minutes")
        
        print(f"best score: {cv.best_score_}")
        print(f"best params: {cv.best_params_}")
        
        # Use best parameters and re-fit on training set
        model.set_params(**cv.best_params_)

        print('Training model on training set using optimal parameters...')
        model.fit(X_train, Y_train)

        # evaluate on test set
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # store model to deploy into production app later
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
