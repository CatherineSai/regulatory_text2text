

Setup

1. create a virtual environment to install the packages in and run the project: 
        python3 -m venv .project_env

2. install necessary packages in the right versions within virtual env.: 
        pip install -r requirements.txt

        If you get a runtime error, try "pip --timeout=1000 install -r requirements.txt"

3. install spacy en_core_web_lg pipeline (https://spacy.io/usage)


--------

Execution 

make sure to load your input .txt files in the corresponding input folder (these should come from the preprocessing script)

review input --> defined_word_lists and adjust according to your regulatory document and realization

run main.py

results will be written to folder results
