import dash

#import dash_bootstrap_components as dbc

# Utilisation de Bulma via un CDN
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']
#external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)#,external_stylesheets=external_stylesheets

app.title = "iMDiNE"
#type_storage="memory"
type_storage="session"



