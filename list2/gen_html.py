import sys
import os
# This script converts a python script to jupyter notebook, executes and saves the notebook, then converts the output to an html file.

# To explain more clearly, here is an example with a script called 'ex1.py' in a folder called 'ex1/':
# py2nb ex1/ex1.py
# jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --ExecutePreprocessor.timeout='python3' --output='ex1_out.ipynb' --execute ex1/ex1.ipynb
# jupyter nbconvert --to html ex1/ex1_out.ipynb

# Python file name
pythonScript = sys.argv[1]
fileName = pythonScript[:-3]
notebookName = fileName + '_out.ipynb'
notebookNameWithoutPath = notebookName.split('/')[-1]

# Command
cmd = 'py2nb ' + pythonScript + ' && jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --ExecutePreprocessor.kernel_name=\'python3\' --output=\'' + notebookNameWithoutPath + '\' --execute ' + fileName + '.ipynb && jupyter nbconvert --to html ' + notebookName

# Execute command
os.system(cmd)
