

Setup

1. create a virtual environment to install the packages in and run the project: 
        python3 -m venv .project_env

2. install necessary packages in the right versions within virtual env.: 
        pip install -r requirements.txt

        If you get a runtime error, try "pip --timeout=1000 install -r requirements.txt"

3. install spacy en_core_web_lg pipeline (https://spacy.io/usage)

4. in a seperate project folder and environment install https://github.com/CatherineSai/text2textPreProcessing 
   This script is necessary for the anaphora resolution (as part of preprocessing), as the chosen algorithm (neuralcoref) is not compatible with Python3.9 and     Spacy3.x. 
--------

Execution 

- upload raw .txt file into the Preprocessing repository (see 4.) and run main.py of that repository

- create an input folder and make sure to load your input .txt files (these should come from the preprocessing script) in the corresponding sub-folder (chech the file_paths script if uncertain about folder sturcture) 

- under input --> defined_word_lists create mapping lists according to your regulatory document and realization

- run main.py

- results will be written to folder "results"
