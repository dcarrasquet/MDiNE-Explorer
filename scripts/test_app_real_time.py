import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import threading
import queue
import os
import pty
import subprocess
import sys

# Crée une application Dash
app = dash.Dash(__name__)

# File d'attente pour stocker les messages de sortie du thread
output_queue = queue.Queue()

# Fonction pour lire la sortie du subprocess et mettre dans la file d'attente
# def read_output(fd):
#     while True:
#         output = os.read(fd, 1024).decode()
#         if not output:
#             break
#         output_queue.put(output.strip())
def read_output(fd):
    buffer = b''
    while True:
        try:
            chunk = os.read(fd, 1024)
            #print("Chunk reussi")
            print("Chunk: ",chunk)
        except OSError:
            break
        buffer += chunk
        #buffer = chunk
        lines = buffer.splitlines()
        #print("Lines: ",lines)
        for line in lines[:-1]:
            output_queue.put(line.decode().strip())
        buffer = lines[-1]
        print("Buffer: ",buffer)
        output_queue.put(buffer.decode().strip())
    if buffer:
        output_queue.put(buffer.decode().strip())
        print("Output queue:",output_queue)

# Fonction exécutée dans un thread
def execute_long_task():
    pid, fd = pty.fork()
    if pid == 0:
        # Processus enfant : exécute le script long_task_script.py
        os.execvp(sys.executable, [sys.executable, 'scripts/long_task_script.py'])
    else:
        # Processus parent : lit la sortie du processus enfant
        read_output(fd)

# Layout de l'application Dash
app.layout = html.Div([
    html.Button('Start Long Task', id='start-button', n_clicks=0),
    html.Div(id='output-area'),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )
])

# Callback pour démarrer la tâche longue dans un thread
@app.callback(
    Output('output-area', 'children'),
    [Input('start-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def start_task(n_clicks, n_intervals):
    if n_clicks > 0:
        # Démarrer la tâche longue dans un nouveau thread
        threading.Thread(target=execute_long_task).start()

    # Récupérer la dernière ligne de la file d'attente
    last_output = ''
    while not output_queue.empty():
        last_output = output_queue.get()

    return html.Div(last_output)
# def start_task(n_clicks, n_intervals):
#     if n_clicks > 0:
#         # Démarrer la tâche longue dans un nouveau thread
#         threading.Thread(target=execute_long_task).start()

#     # Lire et afficher les messages de la file d'attente
#     output_messages = []
#     while not output_queue.empty():
#         output_messages.append(html.Div(output_queue.get()))

#     return output_messages

if __name__ == '__main__':
    app.run_server(debug=True)
