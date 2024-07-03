import subprocess
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import psutil
import sys
import time
import threading
import os
import pty
from dash.exceptions import PreventUpdate
import re
import json

app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='process-store'),
    html.Button('Start Process', id='start-button'),
    html.Button('Kill Process', id='kill-button'),
    html.Div(id='output'),
    html.Div(id='output2'),
    dcc.Interval(id='interval-component', interval=100, n_intervals=0),
    #dcc.Progress(id='progress-bar', value=0, max=100),
    html.Progress(id="progress-bar", value="0", max="100", style={"width": "100%", "height": "30px"}),
    html.Div(id='chains'),
    html.Div(id='divergences'),
    html.Div(id='time')
])

@app.callback(
    Output('process-store', 'data'),
    Input('start-button', 'n_clicks')
)
def start_process(n_clicks):

    if n_clicks!=None and n_clicks>=1:
        master_fd, slave_fd = pty.openpty()

        def write_output_to_file(fd):
            with os.fdopen(fd, 'r') as output:
                for line in output:
                    data=extract_info(line)
                    with open("data/dash_app/session_44/output_model.json", 'w') as file:
                        json.dump(data, file, indent=4)

        info_file="{'filename': 'data/dash_app/session_44/crohns-numeric-tsv.tsv', 'session_folder': 'data/dash_app/session_44/', 'nb_rows': 100, 'nb_columns': 11, 'covar_start': 2, 'covar_end': 5, 'taxa_start': 6, 'taxa_end': 11, 'reference_taxa': 'otu.counts.ref', 'phenotype_column': None, 'first_group': None, 'second_group': None, 'filter_zeros': None, 'filter_dev_mean': None, 'status-run-model': 'not-yet', 'parameters_model': {'beta_matrix': {'apriori': 'Normal', 'parameters': {'alpha': 1, 'beta': 1}}, 'precision_matrix': {'apriori': 'Lasso', 'parameters': {'lambda_init': 10}}}}"
        cmd=[sys.executable, 'scripts/mdine/MDiNE_model.py',info_file]

        process = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,text=True, close_fds=True)
        
        os.close(slave_fd)
        threading.Thread(target=write_output_to_file, args=(master_fd,)).start()
        info = os.fstat(master_fd)
        # Utilisez psutil pour obtenir le PID du processus associé
        print("PID Info: ",info)
        # while True:
        #     read_output(master_fd)
        print("Parent PID ",os.getpid())
        print("Pid process:",process.pid)
        return process.pid

    return None

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'children'),
    Output('chains', 'children'),
    Output('divergences', 'children'),
    Output('time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_progress(n):
    data = read_file()
    percentage = data.get('percentage', 0)
    chains = data.get('chains', 'N/A')
    divergences = data.get('divergences', 'N/A')
    time_remaining = data.get('time', 'N/A')

    return (str(percentage), f"{percentage}%", f"Chains: {chains}", f"Divergences: {divergences}", f"Time: {time_remaining}")


def read_file(file_path="data/dash_app/session_44/output_model.json"):
    try:
        with open(file_path, 'r') as file:
            data=json.load(file)
            
            #print('COntent: ',data)
            #print(type(data))
        if data!=None:
            #print("Je passe par la")
            return data
        else:
            return {}
    except:
        return {}


def extract_info(output):
    #print("Output: ",output)
    match = re.search(
        r"Sampling (\d+) chains, (\d+) divergences.*?(\d+)%.*?(\d+:\d+:\d+)",
        output
    )
    if match:
        return {
            "chains": int(match.group(1)),
            "divergences": int(match.group(2)),
            "percentage": int(match.group(3)),
            "time": match.group(4)
        }
    return None


# def start_process(n_clicks):
#     if n_clicks==1:
#         print("Je suus al")
#         pid, fd = pty.fork()
#         pid_child=None
#         if pid == 0:
#             # Code pour l'enfant (processus)
#             info_file="{'filename': 'data/dash_app/session_44/crohns-numeric-tsv.tsv', 'session_folder': 'data/dash_app/session_44/', 'nb_rows': 100, 'nb_columns': 11, 'covar_start': 2, 'covar_end': 5, 'taxa_start': 6, 'taxa_end': 11, 'reference_taxa': 'otu.counts.ref', 'phenotype_column': None, 'first_group': None, 'second_group': None, 'filter_zeros': None, 'filter_dev_mean': None, 'status-run-model': 'not-yet', 'parameters_model': {'beta_matrix': {'apriori': 'Normal', 'parameters': {'alpha': 1, 'beta': 1}}, 'precision_matrix': {'apriori': 'Lasso', 'parameters': {'lambda_init': 10}}}}"
#             cmd=[sys.executable, 'scripts/mdine/MDiNE_model.py',info_file]
#             result=subprocess.run(cmd)
#             #process= psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
#             #process=psutil.Popen(cmd,shell=False)
#             pid_child=result.pid
#             print("PID: ",result.pid)
#             print("Coucuc")
#         else:
#             print("Non non non", os.getpid())
#             print("On sait jamais",pid_child)
#             # Code pour le parent (monitoring)
#             info = os.fstat(fd)
#             # Utilisez psutil pour obtenir le PID du processus associé
#             print("PID Info: ",info)
#             output = os.fdopen(fd)
#             while True:
#                 line = output.readline()
#                 if not line:
#                     break
#                 print("Coucouc: ",line.strip())

@app.callback(
    Output('output2', 'children'),
    Input('kill-button', 'n_clicks'),
    State('process-store', 'data'),
)
def kill_process(n_clicks,data):
    if n_clicks==0:
        processes = list(psutil.process_iter())

        print("Current PID: ",os.getpid())

        children=["Current PID: ",os.getpid()]

        # Affichage des détails de chaque processus
        for process in processes:
            try:
                # Récupérer les informations de base sur le processus
                process_info = process.as_dict(attrs=['pid', 'name', 'username'])

                children.append(f"PID: {process_info['pid']}, Nom: {process_info['name']}, Utilisateur: {process_info['username']}")

                # Afficher les informations du processus
                #print(f"PID: {process_info['pid']}, Nom: {process_info['name']}, Utilisateur: {process_info['username']}")
            except psutil.NoSuchProcess:
                pass  # Ignorer les processus qui ne sont plus actifs
        return children


    if n_clicks!=None and n_clicks>=1:
        print("Nouveau pid:",os.getpid())
        pid_to_kill=int(data)
        try:
            process = psutil.Process(pid_to_kill)
            process.terminate()  # Terminer le processus
            print(f"Processus avec PID {pid_to_kill} terminé avec succès.")
        except psutil.NoSuchProcess:
            print(f"Processus avec PID {pid_to_kill} n'existe pas.")
        except psutil.AccessDenied:
            print(f"Accès refusé pour terminer le processus avec PID {pid_to_kill}.")

        return "Coucou"
    else:
        raise PreventUpdate


# def start_process(n_clicks):
#     pid=None
#     print("nclicks",n_clicks)
#     # if n_clicks==0:
#     #     cmd="python scripts/mdine/MDiNE_model.py {'filename': 'data/dash_app/session_44/crohns-numeric-tsv.tsv', 'session_folder': 'data/dash_app/session_44/', 'nb_rows': 100, 'nb_columns': 11, 'covar_start': 2, 'covar_end': 5, 'taxa_start': 6, 'taxa_end': 11, 'reference_taxa': 'otu.counts.ref', 'phenotype_column': None, 'first_group': None, 'second_group': None, 'filter_zeros': None, 'filter_dev_mean': None, 'status-run-model': 'not-yet', 'parameters_model': {'beta_matrix': {'apriori': 'Normal', 'parameters': {'alpha': 1, 'beta': 1}}, 'precision_matrix': {'apriori': 'Lasso', 'parameters': {'lambda_init': 10}}}}"
#     #     info_file="{'filename': 'data/dash_app/session_44/crohns-numeric-tsv.tsv', 'session_folder': 'data/dash_app/session_44/', 'nb_rows': 100, 'nb_columns': 11, 'covar_start': 2, 'covar_end': 5, 'taxa_start': 6, 'taxa_end': 11, 'reference_taxa': 'otu.counts.ref', 'phenotype_column': None, 'first_group': None, 'second_group': None, 'filter_zeros': None, 'filter_dev_mean': None, 'status-run-model': 'not-yet', 'parameters_model': {'beta_matrix': {'apriori': 'Normal', 'parameters': {'alpha': 1, 'beta': 1}}, 'precision_matrix': {'apriori': 'Lasso', 'parameters': {'lambda_init': 10}}}}"
#     #     cmd1=[sys.executable, 'scripts/mdine/MDiNE_model.py',info_file]
#     #     p = psutil.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
#     #     print('PID: ', p.pid)
#     #     return p.pid
#     if n_clicks!=None and n_clicks>=1:
#         buffer = b''
#         info_file="{'filename': 'data/dash_app/session_44/crohns-numeric-tsv.tsv', 'session_folder': 'data/dash_app/session_44/', 'nb_rows': 100, 'nb_columns': 11, 'covar_start': 2, 'covar_end': 5, 'taxa_start': 6, 'taxa_end': 11, 'reference_taxa': 'otu.counts.ref', 'phenotype_column': None, 'first_group': None, 'second_group': None, 'filter_zeros': None, 'filter_dev_mean': None, 'status-run-model': 'not-yet', 'parameters_model': {'beta_matrix': {'apriori': 'Normal', 'parameters': {'alpha': 1, 'beta': 1}}, 'precision_matrix': {'apriori': 'Lasso', 'parameters': {'lambda_init': 10}}}}"
#         cmd1=[sys.executable, 'scripts/mdine/MDiNE_model.py',info_file]
#         process = psutil.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,text=True)

#         # stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
#         # stdout_thread.start()
#         fd=process.stdout.fileno()
#         while True:
#             print("Coucoucee")
            
#             print("J'ai essayé")
#             chunk = os.read(fd, 1024)
#             #chunk = os.read(fd, 2048)
#             print("chunk:",chunk)
            
#             buffer += chunk
#             lines = buffer.splitlines()
#             for line in lines[:-1]:
#                 decoded_line = line.decode().strip()
#                 #print('decoded line:', str(decoded_line))
#                 print("Adbien: ",decoded_line)
#                 # decoded_line = remove_ansi_escape_sequences(buffer.decode(errors='replace').strip())
#                 # print("Extracted info:",extract_information(decoded_line))
#                 # extracted_info = extract_info(decoded_line)
#                 # if extracted_info:
#                 #     # Store the info in dcc.Store
#                 #     app.callback(
#                 #         Output('progressbar-info', 'data'),
#                 #         Input('interval-component', 'n_intervals')
#                 #     )(lambda n: extracted_info)
#             buffer = lines[-1]

#         while False:
            

#             # Read standard output data.
#             while True:
#                 process.stdout.flush()
#                 output = process.stdout.read(1)  # Lire un caractère à la fois
#                 print("Output: ",output)
#                 if output == '' and process.poll() is not None:
#                     break
#                 if output:
#                     sys.stdout.write(output)
#                     sys.stdout.flush()

#             # Read standard error data.
#             while True:
#                 error = process.stderr.read(1)  # Lire un caractère à la fois
#                 if error == '' and process.poll() is not None:
#                     break
#                 if error:
#                     sys.stderr.write(error)
#                     sys.stderr.flush()

#             # If the process is complete - exit loop.
#             if process.poll() is not None:
#                 break


#         # while True:
#         #     time.sleep(5)

#         #     # Read standard output data.
#         #     for stdout_line in iter(process.stdout.readline, ""):

#         #         print("Ligne normale: ",stdout_line)

#         #     # Read standard error data.
#         #     for stderr_line in iter(process.stderr.readline, ""):

#         #         # Display standard error data.
#         #         print("Ligne error: ",stderr_line)
#         #         #sys.stderr.write(stderr_line)

#         #     # If the process is complete - exit loop.
#         #     if process.poll() != None:
#         #         break
#     return None

@app.callback(
    Output('output', 'children'),
    Input('process-store', 'data')
)
def get_process_output(pid):
    if pid:
        try:
            with open(f'/proc/{pid}/fd/1', 'r') as f:  # stdout file descriptor
                output = f.read()
            return output
        except FileNotFoundError:
            return "Process not found or finished."
    return "No process started."

if __name__ == '__main__':
    app.run_server(debug=True)
