from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)

app.layout = html.Div([
    html.Button("Open Modal", id="open-modal-button"),
    html.Div(
        id="modal",
        style={"display": "none", "position": "fixed", "z-index": "1", "left": "0", "top": "0", "width": "100%", "height": "100%", "overflow": "auto", "background-color": "rgb(0,0,0)", "background-color": "rgba(0,0,0,0.4)"},
        children=[
            html.Div(
                style={"background-color": "#fefefe", "margin": "15% auto", "padding": "20px", "border": "1px solid #888", "width": "80%"},
                children=[
                    html.Span("Close", id="close-modal-button", style={"float": "right", "cursor": "pointer"}),
                    html.H2("This is a Modal"),
                    html.P("Some more content in the modal...")
                ]
            )
        ]
    )
])

@app.callback(
    Output("modal", "style"),
    [Input("open-modal-button", "n_clicks"), Input("close-modal-button", "n_clicks")],
    [State("modal", "style")]
)
def toggle_modal(n_open, n_close, style):
    print("Je suis appelé")
    if n_open or n_close:
        if style["display"] == "none":
            style["display"] = "block"
        else:
            style["display"] = "none"
    return style

if __name__ == "__main__":
    app.run_server(debug=True)
