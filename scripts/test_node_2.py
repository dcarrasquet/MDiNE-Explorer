import os
import pty
import subprocess
import sys

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

if __name__ == '__main__':
    execute_long_task()
