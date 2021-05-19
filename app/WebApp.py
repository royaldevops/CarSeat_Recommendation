# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:59:07 2021

@author: aaron
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
#from google.cloud import bigquery
#from google.oauth2 import service_account
#import json
#import tempfile
#import plotly.express as px
import pandas as pd

import re 
#from datetime import datetime, timedelta
import pickle
#import numpy as np

from nltk.stem import PorterStemmer

###############################
# start the App
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Car Seat Recommendation'
server = app.server


with open('data_master.pickle', 'rb') as f:
    data_master = pickle.load(f)
with open('data_light.pickle', 'rb') as f:
    data_light = pickle.load(f)

keywords = ['Lightweight', 'Comfortable', 'Stylish', 'Safe', 'Newborn-Fridendly', 'Easy-to-Install', 'Easy-to-Clean', 'Easy-to-Adjust', 'Sturdy', 'Spacious']
keywords_raw = ['lightweight', 'comfortable', 'stylish', 'safe', 'newborn', 'install', 'clean', 'adjust', 'sturdy', 'spacious']

def top_3(preference):
    loc = []
    for i, j in enumerate(keywords):
        if j in preference:
            loc.append(i+1)                
    top3 = data_master[['asin', 'product'] + [f'keyword{i}_sentiment' for i in loc]]  
    top3['sum_sentiment'] = data_master[[f'keyword{i}_sentiment' for i in loc]].sum(axis=1)
    top3 = top3.sort_values(['sum_sentiment'], ascending = False, ignore_index = True)
    top3 = top3.iloc[:3,:]  
    top3_w_review = data_light.merge(top3, how = 'right', on = 'asin')

    top3_w_review['feature'] = '' 
    for i in loc:
        top3_w_review['feature'] = top3_w_review.apply(lambda x: f"{x['feature']}, {keywords[i-1]}" if x[f'keyword{i}_sentiment_rank'] <= 3 else x['feature'], axis = 1)
    top3_w_review = top3_w_review[top3_w_review['feature']!='']
    top3_w_review['feature'] = top3_w_review['feature'].str.replace(',','',1).str.strip()

    return top3_w_review, loc

stemmer= PorterStemmer()

def bold_text(review, loc):
    review = review.replace('.', '. ').replace(',', ', ').replace('?', '? ').replace('!', '! ').replace(':', ': ').replace('(', ' (').replace('*', '')
    review = ' '.join(review.split())
    review_bold = ""
    for i in review.split():
        if re.findall('|'.join([stemmer.stem(keywords_raw[i-1]) for i in loc]), stemmer.stem(i)):       
            review_bold = review_bold + ' ' + '_**'+ i + '**_'
        else:
            review_bold = review_bold + ' ' + i
    return review_bold

def prepare_tables(df, loc, asin_n):
    ##########
    # prepare asin table
    prod1_asin = df[df['asin']==asin_n][['asin', 'product'] + [f'keyword{i}_sentiment' for i in loc]].drop_duplicates()
    for i in loc:
        prod1_asin[f'keyword{i}_sentiment'] = prod1_asin.apply(lambda x: str(x[f'keyword{i}_sentiment']) + ' / 5'   , axis =1)
    dict1 = {'ASIN': [html.A(html.P('Amazon Link'), href=f'https://www.amazon.com/dp/{prod1_asin.asin.values[0]}', target="_blank")], 
            'Product':  [prod1_asin['product'].values[0]]}

    dict1.update({f'Rating for {keywords[i-1]}': [prod1_asin[f'keyword{i}_sentiment'].values[0]] for i in loc})
    prod1_asin = pd.DataFrame(dict1, index = [0])

    ##########
    # prepare review table
    df_review = df[df['asin']==asin_n][['feature', 'review']]
    df_review = df_review.sort_values(['feature'])
    reviews = df_review.review.values.tolist()
    reviews = [dcc.Markdown(bold_text(i, loc)) for i in reviews]
    dict_review = {'Top Customer Reviews': reviews}
    df_review = pd.DataFrame(dict_review)

    #######
    # prepare bdc.table
    table = dbc.Table.from_dataframe(prod1_asin, striped=True, bordered=True, hover=True)
    table_review = dbc.Table.from_dataframe(df_review, striped=True, bordered=True, hover=True)

    return table, table_review
############################################################
# prepare the layout
app.layout = html.Div([
    html.H1(children='Baby Car Seat Recommendation System'),
    html.Div(children='Baby Car Seat Recommendation System Based on Amazon Product Reviews and Customer Preferences'),
    html.Div(children='Developer: Aaron Zhu'),
    html.Br(),
    
    dbc.Checklist(
        options=[{"label": x, "value": x} for x in keywords],
        value=[],
        labelStyle={'display': 'inline-block'},
        labelCheckedStyle={"color": "red"},
        inline=True, # arrange list horizontally
        id="features-input",        
    ),
    html.Br(),
    dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary"),
    html.Br(),

    html.Div(id='more_than_3_features'),
    html.Br(),html.Br(),

    dbc.Tabs([
        dbc.Tab(label='Product One', 
            children = [
                html.Div(id='prod1_table'),
                html.Div(id='prod1_review')
            ]
        ),

        dbc.Tab(label='Product Two', 
            children = [
                html.Div(id='prod2_table'),
                html.Div(id='prod2_review')
            ]
        ),
        dbc.Tab(label='Product Three', 
            children = [
                html.Div(id='prod3_table'),
                html.Div(id='prod3_review')
            ])
    ])
], style = {'padding': '20px'})

#############
# create callback function
@app.callback(
    [Output('more_than_3_features', 'children'),
    Output('prod1_table', 'children'),
    Output('prod1_review', 'children'),
    Output('prod2_table', 'children'),
    Output('prod2_review', 'children'),
    Output('prod3_table', 'children'),
    Output('prod3_review', 'children')],
    [Input('submit-val', 'n_clicks'),
    State('features-input', 'value')])
def recommendation_system(n_clicks, features):
    if len(features)==0:
        return 'Note: Please select up to 3 features.', {}, {}, {}, {}, {}, {}
    elif len(features)>3:
        return 'Note: Please select between 1 to 3 features.', {}, {}, {}, {}, {}, {}
    else:
        df, loc = top_3(features)
        keywords_sel = [keywords[i-1] for i in loc]
        asins = df['asin'].unique().tolist()
        print(keywords_sel, asins)    

        prod1, prod1_review = prepare_tables(df, loc, asins[0])
        prod2, prod2_review = prepare_tables(df, loc, asins[1])
        prod3, prod3_review = prepare_tables(df, loc, asins[2])

        return {}, prod1, prod1_review, prod2, prod2_review, prod3, prod3_review

# recommend if user only select one/ two feature [done]
# use dcc.Markdown to bold features [done]
# add second and third product [done]
# add average amazon general rating
# add rating per review
# remove irrelevant product. for example, B07BVWFPKZ, B00D45BF6G, B07D2SK89Y [done]

#############################
print('****************************************************')
# run the app 
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    