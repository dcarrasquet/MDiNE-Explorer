import sys
import time

import subprocess
import threading

# Define the command you want to run
command = ["ping", "example.com"]

# Create a Popen object to run the command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Define a function to read and process stdout
def capture_stdout():
    while True:
        line = process.stdout.readline()
        if not line:
            break  # No more output, exit the loop
        print("stdout:", line.strip())  # Process and print the line

capture_stdout()

# # Start a separate thread to capture stdout
# stdout_thread = threading.Thread(target=capture_stdout)
# stdout_thread.start()

# # Wait for the process to complete and get the return code
# return_code = process.wait()
# stdout_thread.join()  # Wait for the stdout thread to finish

# # Print the final return code
# print("Return Code:", return_code)
 