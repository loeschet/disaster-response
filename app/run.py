import pandas as pd
from os.path import join, dirname
import sys
sys.path.insert(0, join(dirname(__file__), "../models"))
from train_classifier import ColumnSelector, tokenize
from flask import Flask
from flask import render_template, request, jsonify
import joblib
import pickle
from sqlalchemy import create_engine

# custom unpickler needed for proper unpickling when deploying using
# gunicorn
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

# load data from SQLite database
file_path = join(dirname(__file__), "../data/DisasterResponse.db")
engine = create_engine(f'sqlite:///{file_path}')
df = pd.read_sql_table('DisasterResponse', engine)
df['tokens'] = df['message'].map(lambda x: tokenize(x))

# load scikit-learn model from pickle file
file_path = join(dirname(__file__), "../models/classifier.pkl")
#model = joblib.load(file_path)
model = CustomUnpickler(open(file_path, 'rb')).load()

def init_routes(app):
    '''
    INPUT:
    app - Flask app
    
    This function simply initializes the routes for our Flask app.
    '''

    # index webpage displays cool visuals and receives user input text for
    # model
    @app.route('/')
    @app.route('/index')
    def index():
        
        # render web page with plotly graphs
        return render_template('master.html')


    # web page that handles user query and displays model results
    @app.route('/go')
    def go():
        # save user input in query
        query = request.args.get('query', '')
        tmp_df = pd.DataFrame({"message": [query]})

        # use model to predict classification for query
        classification_labels = model.predict(tmp_df)[0]
        classification_results = dict(zip(df.columns[4:-1],
                                          classification_labels))

        # This will render the go.html Please see that file. 
        return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
        )

def init_app():
    '''
    This function initializes our Flask app. Note that to spice up the data
    visualization a bit, I integrated a plotly dash app (and not just a simple
    plotly plot) into the existing Flask app. The respective dash application
    is launched from the file "dashapp.py". The app itself is displayed using
    an iframe (see bottom of "master.html" file)
    '''
    app = Flask(__name__)
    with app.app_context():
        init_routes(app)

        # Import Dash application
        try:
            from .dashapp import init_dashboards
        except ImportError:
            from dashapp import init_dashboards
        
        app = init_dashboards(app)

        return app

def run_app(host=None, port=None, debug=None):
    # initialize and run Flask app
    app = init_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(host='0.0.0.0', port=3001, debug=True)
