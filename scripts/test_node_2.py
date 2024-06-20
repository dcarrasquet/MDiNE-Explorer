import os
import pty
import subprocess
import sys
import threading
import time

def execute_long_task():
    def read(fd):
        while True:
            output = os.read(fd, 1024).decode()
            if not output:
                break
            print(output, end='')

    pid, fd = pty.fork()
    if pid == 0:
        # This is the child process
        os.execvp(sys.executable, [sys.executable, 'scripts/long_task_script.py'])
    else:
        # This is the parent process
        read(fd)

def main():
    print("Début du programme principal")
    
    # Démarrage de la tâche longue dans un thread
    thread = threading.Thread(target=execute_long_task)
    thread.start()
    
    # Continuer avec d'autres tâches dans le programme principal
    time.sleep(3)
    print("Le programme principal continue son exécution pendant que la tâche tourne en arrière-plan")

    # Attendre que le thread de la tâche longue se termine si nécessaire
    thread.join()

    print("Fin du programme principal")

if __name__ == '__main__':
    # print("Debut de la tache")
    # execute_long_task()
    # print("Fin de la tache")
    main()
