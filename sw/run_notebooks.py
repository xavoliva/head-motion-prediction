"""
Python script to run notebooks in NOTEBOOK_FILENAMES sequentially
"""

import subprocess
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_FILENAMES = [
    "final_seq2seq_teacher_forcing_1-64-test.ipynb"
]

for notebook_filename in NOTEBOOK_FILENAMES:
    subprocess.Popen(["jupyter", "notebook"])
    time.sleep(3)
    print("\n\nExecuting", notebook_filename+"...")

    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    try:
        out = ep.preprocess(nb, {'metadata': {'path': ''}})
    except ValueError:
        msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
        msg += 'See notebook "%s" for the traceback.' % notebook_filename
        print(msg)
        raise
    finally:
        nbformat.write(nb, open(notebook_filename, mode='wt'))
        print("Execution finished")
        subprocess.run(["jupyter", "notebook", "stop", "8888"])
