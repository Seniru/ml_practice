pyenv global 3.7.6
python -m pip install -r requirements.txt
jupyter notebook --NotebookApp.allow_origin=\'$(gp url 8888)\'
