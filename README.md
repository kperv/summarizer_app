# A web application to summarize a text into several sentences. 


To run the web part (no model connected yet) run *run.sh* from the *app* folder: `cd app && ./run.sh`. 
The web interface will be available on *localhost:8080* in your browser.

The model is built around BART with attempt to try building the HieBart architecture.
https://aclanthology.org/2021.naacl-srw.20.pdf

As a first step the basic architecture is created and is being tested.

In the data_preprocessing.py module the MLSUM dataset is loaded and tokenized.
In model.py the data is organized for training. 
