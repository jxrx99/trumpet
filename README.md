## TRUMPet 

_Generating Fake Tweets by Trump has never been easier_

**Documentation**: [https://]

TRUMPet is a trained model that attempts to emulate the tweeting style of the tweets it is trained on. 
In this case, our data set is Trump Tweet.
In our product, we invite you to guess which tweets were real and which were generated.


We are utilising the following open source tools to power the core of TRUMPet,
- **[Tensorflow](https://www.tensorflow.org/api_docs/python/)**: TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks
- **[GloVe](https://nlp.stanford.edu/projects/glove/)**: GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.


#### Installation and requirements

**Python**
- 3.6 and above

**Tensorflow**

- v2.3.0

**Virtual environment**


- Install the required dependencies 
```
conda env create -f conda.yml
```


#### Folder directory

```
team5
|_src
  |_templates 
  |_datapipeline
  |_experiment
  |_modelling
  |_app.py
|_ data
|_ notebooks
|_ conda.yml
|_ requirements.txt
|_ polyaxon

```
- src: contains all the py files to run the program
- data: contains trump.csv (raw data) **[Trump Tweets Dataset](https://www.kaggle.com/austinreese/trump-tweets)**.
- notebooks: contains the notebook for testing purpose before writing the py files
- conda.yml: contains all the neccesary libraries to run the app
- polyaxon: allows the running on the experiment on the polyaxon

## Running the app locally

#### Flask app

You can test if this app works locally by running 

```
python -m src.app
```







#### Contributors

Jin Howe Teo, Ce Zheng Neo, Muhammad Jufri Bin Ramli, Raymond Liu


##### Acknowledgements 

Special thanks to our mentor Kenneth Wang and AIAP