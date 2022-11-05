from dash import Dash, html, dcc, Input, Output
import pandas as pd
from plotly.graph_objects import Bar, Figure
from os.path import join, dirname
from sqlalchemy import create_engine
import numpy as np
from itertools import islice
import sys
sys.path.insert(0, join(dirname(__file__), "../models"))
from train_classifier import tokenize

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def get_data():
    # load data from SQLite database into pandas dataframe
    file_path = join(dirname(__file__), "../data/DisasterResponse.db")
    engine = create_engine(f'sqlite:///{file_path}')
    df = pd.read_sql_table('DisasterResponse', engine)
    df['tokens'] = df['message'].map(lambda x: tokenize(x))

    # create "result" dictionary containing for each disaster category another
    # dictionary, whose keys are all tokens from tweets in that
    # category and whose values are the respective counts (number of
    # occurrences) of these tokens
    res_dict = {}
    classes = df.columns[4:-1]
    for cls in classes:
        tmp_df = df[df[cls]==1]
        res_dict[cls] = {}
        for i, dat in tmp_df['tokens'].items():
            for wrd in dat:
                if wrd in res_dict[cls].keys():
                    res_dict[cls][wrd] += 1
                else:
                    res_dict[cls][wrd] = 1

    return df, res_dict

def plot_genres(df):
    '''
    Plot distribution of message genres
    '''

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    bar = Bar(x=genre_names, y=genre_counts)
    layout = {'title': 'Distribution of Message Genres',
              'yaxis': {'title': "Count"},
              'xaxis': {'title': "Genre"}}
    
    fig_genres = Figure(data=bar, layout=layout)
    
    return fig_genres

def init_dashboards(server):
    '''
    Initialize plotly dash app. The "server" argument will be our original
    Flask app that we want to integrate our plotly dash visualizations into.
    '''
    
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    dash_app = Dash(server=server, routes_pathname_prefix="/dashapp/",
                    external_stylesheets=external_stylesheets)

    # load data
    df, res_dict = get_data()

    # define plotly dash layout
    dash_app.layout = html.Div([
        html.Div([
            dcc.Graph(figure=plot_genres(df))
        ], style={'display': 'inline-block', 'width': '100%'}),
        
        html.Div([
            dcc.Dropdown(options=[
                {'label': cls, 'value': cls} for cls in df.columns[4:-1]
                ], value='floods',
                        id='crossfilter-columns')
        ], style={'width': '100%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(
                id='crossfilter-bar-chart',)
        ], style={'width': '100%', 'display': 'inline-block',
                  'padding': '0 20'}),
        
        html.Div([
            dcc.Graph(
                id='crossfilter-corr-chart',)
        ], style={'width': '100%', 'display': 'inline-block',
                  'padding': '0 20'})
    ])
    
    # initialize callbacks for updating plotly plots based on dropdown
    # disaster category choice
    init_callbacks(dash_app, df, res_dict)
    
    return dash_app.server

def init_callbacks(dash_app, df, res_dict):

    # This function will display the top five tokens for a disaster category of
    # choice (as chosen by a drop-down list of all available categories)
    # as a horizontal bar chart
    @dash_app.callback(
        Output('crossfilter-bar-chart', 'figure'),
        Input('crossfilter-columns', 'value'))
    def update_wordcounts(column_name):
        sorted_dict = dict(sorted(res_dict[column_name].items(),
                                key=lambda item: item[1], reverse=True))
        
        top_five = np.array(take(5, sorted_dict.items()))
        bar = Bar(x=top_five[:, 1].astype(int), y=top_five[:, 0],
                  orientation='h')

        layout = {'title': ('Distribution of most occurring tokens '
                            f'for {column_name} disasters'),
                'yaxis': {'title': "Token"},
                'xaxis': {'title': "Count"}}
        
        fig_wordcounts = Figure(data=bar, layout=layout)
        
        return fig_wordcounts

    # This function displays the "correlation" of a user-specified disaster
    # category (using a drop-down list) with all other categories. This means
    # that for each true example in that category, it counts how often each of
    # the other categories are also true and plots that distribution of counts
    # as a bar chart.
    @dash_app.callback(
        Output('crossfilter-corr-chart', 'figure'),
        Input('crossfilter-columns', 'value'))
    def update_correlations(column_name):
        tmp_df = df[df.columns[4:-1]]
        tmp_df = tmp_df[tmp_df[column_name] == 1]
        tmp_df = tmp_df.drop(columns=[column_name])
        res = tmp_df.sum()

        bar = Bar(x=res.index, y=res.values)
        layout = {'title': ('Distribution of common labels '
                            f'for {column_name} disasters'),
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Disaster type"}}
        
        fig_correlation = Figure(data=bar, layout=layout)
        
        return fig_correlation