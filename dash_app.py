import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import base64

# Utilisation de Bulma via un CDN
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

#server = app.server

uploaded_file_name = None

# Styles pour la barre de navigation
nav_style = {
    'padding': '10px'
}

tab_style = {
    'backgroundColor': '#4A90E2',
    'color': 'white',
    'padding': '10px',
    'border': 'none',
    'textAlign': 'center',
    'width': '150px',  # Largeur élargie pour les onglets
    'borderRadius': '10px',
    'marginRight': '10px'
}

selected_tab_style = {
    'backgroundColor': '#341f97',
    'color': 'white',
    'padding': '10px',
    'border': 'none',
    'textAlign': 'center',
    'width': '150px',  # Largeur élargie pour les onglets sélectionnés
    'borderRadius': '10px',
    'marginRight': '10px'
}

app.layout = html.Div([
    # Barre de navigation
    html.Nav(style=nav_style, className="navbar", children=[
        html.Div(className="navbar-brand"),
        html.Div(className="navbar-menu", style={'justifyContent': 'center', 'width': '100%'}, children=[
            dcc.Tabs(id="tabs-example", value='tab-data', style={'width': '100%'}, children=[
                dcc.Tab(label='Home', value='tab-home', style=tab_style, selected_style=selected_tab_style),
                dcc.Tab(label='Data', value='tab-data', style=tab_style, selected_style=selected_tab_style),
                dcc.Tab(label='Model', value='tab-model', style=tab_style, selected_style=selected_tab_style),
                dcc.Tab(label='Visualization', value='tab-visualization', style=tab_style, selected_style=selected_tab_style),
                dcc.Tab(label='Export results', value='tab-export', style=tab_style, selected_style=selected_tab_style)
            ])
        ])
    ]),

    # Contenu principal
    html.Div(id='tabs-content', className="container")
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-home':
        return html.Div([
            html.H3('Welcome to the Home Page'),
            html.Div([
                html.P('The aim is to provide an easy-to-use graphical interface for estimating differential co-occurrence networks using the MDiNE model [ref].'),
                html.P('''MDiNE is a Bayesian hierarchical model that estimates species interactions from a table of counts and a table of covariates. 
It is possible to modify aprioris on various model parameters from a predefined list. 
A tutorial is available at: https://example_mdine.com'''),
                html.Img(src='/assets/co_occurence_networks.png', style={'width': '50%','display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
    
    ])
        ])
    elif tab == 'tab-data':
        return html.Div([
            html.H3('Data Page'),
            html.Div(style={'width': '70%', 'display': 'inline-block','clear': 'both', 'borderRight': '3px solid #ccc'}, children=[
        # Contenu de la partie gauche
        dcc.Store(id='upload-status', storage_type='session'),
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Permet d'accepter un seul fichier à la fois
        multiple=False,
        accept='.csv'
    ),
    html.Div(id='output-data-upload')
            ]),
    html.Div(style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
        # Contenu de la partie droite
        html.H1("Partie Droite"),
        # Ajoutez ici les composants de votre partie droite
    ]),
    html.Div(style={'clear': 'both', 'borderLeft': '1px solid #ccc'}),
])
    elif tab == 'tab-model':
        return html.Div([
            html.H3('Model Page')
        ])
    elif tab == 'tab-visualization':
        return html.Div([
            html.H3('Visualization Page'),
            dcc.Tabs(id="subtabs", value='subtab-cooccurrence', children=[
                dcc.Tab(label='Co-occurrence networks', value='subtab-cooccurrence'),
                dcc.Tab(label='Performance metrics', value='subtab-performance')
            ]),
            html.Div(id='subtabs-content')
        ])
    elif tab == 'tab-export':
        return html.Div([
            html.H3('Export Results Page')
        ])

@app.callback(Output('subtabs-content', 'children'),
              [Input('subtabs', 'value')])
def render_subcontent(subtab):
    if subtab == 'subtab-cooccurrence':
        fig = px.scatter(px.data.iris(), x='sepal_width', y='sepal_length', color='species')
        return html.Div([
            html.H4('Co-occurrence networks'),
            dcc.Graph(figure=fig)
        ])
    elif subtab == 'subtab-performance':
        return html.Div([
            html.H4('Performance metrics'),
            html.P('Metrics content goes here.')
        ])
    
# @app.callback(Output('output-data-upload', 'children'),
#               [Input('upload-data', 'contents'), Input('upload-data', 'filename')],
#               prevent_initial_call=False)

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'), Input('upload-data', 'filename')],
              prevent_initial_call=True)
def update_output(contents,filename):
    global uploaded_file_name
    if contents is not None:
        # Récupérer les données et le nom du fichier depuis les métadonnées
        print("Filename: ",filename)

        uploaded_file_name = filename
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        #filename = 'uploaded_file.txt'  # Nom de fichier par défaut
        for content in content_type.split(';'):
            print("Content: ",content)
            if 'filename=' in content:
                filename = content.split('=')[1].strip('"')
        # Enregistrer le fichier sur le serveur avec le même nom que celui déposé
        with open(os.path.join('data/test_app', filename), 'wb') as f:
            f.write(decoded)
        return html.Div([
            html.H5('File successfully downloaded : {}'.format(filename))
        ])


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)
