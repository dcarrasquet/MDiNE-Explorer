# Importation des bibliothèques nécessaires
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Définition de la mise en page de l'application
app.layout = html.Div([
    html.H1("Exemple d'application Dash"),
    html.Label("Sélectionnez une couleur :"),
    dcc.Dropdown(
        id='dropdown-color',
        options=[
            {'label': 'Rouge', 'value': 'red'},
            {'label': 'Bleu', 'value': 'blue'},
            {'label': 'Vert', 'value': 'green'}
        ],
        value='blue'
    )
])

# Exécution de l'application
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)