import dash
from dash import dcc, html, Input, Output, State
import dash_daq as daq

app = dash.Dash(__name__)

type_storage="session"

app.layout = html.Div([
    html.Div(id='color-square', style={'width': '100px', 'height': '100px', 'background-color': '#0000FF'}),
    daq.ColorPicker(
        id='color-picker',
        label='Pick a color',
        value=dict(hex='#0000FF'),
        style={'display': 'none'}
    ),
    dcc.Checklist(options=[{'label': '', 'value': 'checked'}],value=[],id='check-separate-data',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
])

@app.callback(
    Output('color-picker', 'style'),
    Output('color-picker', 'value'),
    Input('color-square', 'n_clicks'),
    State('color-picker', 'value'),
    prevent_initial_call=True
)
def display_color_picker(n_clicks, color_value):
    if n_clicks%2==1:
        return {'display': 'block'}, color_value
    else:
        return {'display': 'none'}, color_value

@app.callback(
    Output('color-square', 'style'),
    Input('color-picker', 'value')
)
def update_square_color(color_value):
    return {'width': '100px', 'height': '100px', 'background-color': color_value['hex']}

if __name__ == '__main__':
    app.run_server(port=8080,debug=True)
