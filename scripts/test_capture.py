import dash
from dash import dcc, html,callback_context
from dash.dependencies import Output, Input
import threading
import subprocess
import time
import sys
import io

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.Button('Run Long Process', id='run-button', n_clicks=0),
    dcc.Store(id='process-pid'),
    html.Div(id='output')
])

# Function to run the long process
def long_running_function(store_pid):
    # Create a subprocess and get its PID
    process = subprocess.Popen(
        ['python3', '-c', 'import time; print("Starting process"); time.sleep(10); print("Process done")'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Store the PID in the dcc.Store component
    callback_context.response.set_data({'process_pid': process.pid})
    
    # Redirect stdout to capture the output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Capture and print the output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    
    # Restore stdout
    sys.stdout = old_stdout

@app.callback(
    Output('process-pid', 'data'),
    [Input('run-button', 'n_clicks')]
)
def run_process(n_clicks):
    if n_clicks > 0:
        thread = threading.Thread(target=long_running_function, args=('process-pid',))
        thread.start()
        return dash.no_update
    return dash.no_update

@app.callback(
    Output('output', 'children'),
    [Input('process-pid', 'data')]
)
def display_output(process_pid):
    if process_pid:
        return f'Process PID: {process_pid}'
    return 'No process running.'

if __name__ == '__main__':
    app.run_server(debug=True)
