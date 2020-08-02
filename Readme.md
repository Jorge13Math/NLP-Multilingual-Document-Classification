# Introduction
This repository aims to show the result and the methodology used to achieve the objectives of the EA technical test. The tasks are based on natural language processing, using Python programming language for their development.

# Getting Started

1. Installation process:
    * Install conda
    * conda create -n testea python=3.6
    * conda activate testea
    * pip install -r requirements.txt
    * python -m nltk.downloader all
    * python -m spacy download en_core_web_sm
    * python -m spacy download fr_core_news_sm
    * python -m spacy download es_core_news_sm
    



1. Software dependencies
    * All python packages needed are included in `requirements.txt`

1. Test the model:
    * If step 1 is OK you can run the line's code below to deploy a server and test the model.
    * Go to **app** folder 
    ```
    cd code\app
    ```
    * Run the following line in cmd
    
    ```
    python server.py
    ```
    * Go to **http://localhost:9000/**.  Be sure the port 9000 is not busy, otherwise the server does not work.

# Methodology
The repository is structured in the following folders:

 * code: You will find the python scripts and a folder called "notebooks".

* data: You will find the folders categorized by documents and the CSV file generated in point 1.

 * templates: You will find the templates for the server.

**Generate Dataset**

1. A dataset was created with all the files that are in the data folder
    * To view the Dataset go to **Generate Dataframe.ipynb** 

**OBJECTIVE 1**: Create a document categorization classifier for the different contexts of the documents

**Pre-process**

To perform pre-processing, a class called "preprocess_data.py" was developed. This class has different methods:

* clean_dataframe: It was used to clean / pre-process the data, where null values, duplicate values, stopwords of the three languages ​​and any other noise such as digits and punctuation marks are eliminated. Also, words with less than three characters were removed.
* To view the other methods go to notebooks: **Pipeline classifier.ipynb**

**Model**

 Classifier:

For this task, the **TFIDF** (Term frequency – Inverse document frequency) was used to transform text into a features to train the model. This method consists of creating a matrix that contains the frequency of the words found in each document.

 * Two neural networks were trained. 
One **MLP** (Multi layer perceptron) and the other **LSTM** (Long short term memory). In the notebook **Pipeline classifier.ipynb** the performance of both models is appreciated, the comparison is made using the precision vs recall curve metric. 

  In this case, by comparing the two models, the LSTM achieved better results for all classes using the metric mentioned before, for this reason it was chosen to make the final model and be tested in the server.

**OBJECTIVE 2**: Perform a topic model analysis on the provided documents

**Pre-process**

To perform pre-processing, the same class called "preprocess_data.py" was used.

**Model**

Topic Modeling:

For this task, was used **LDA** (Latent Dirichlet Allocation) to find different topics in the dataset.
To reproduce the results, go to **Topic Modeling  LDA.ipynb**

**CONCLUSION**


In accordance with the objectives set out in the test, two types of neural networks were used to classify the text regardless of the language. The results showed that the optimal way to classify this type of text is to use LSTM because its implementation predicts sequences and in this case, the text had different sequences for each language.


It should be noted that the dataset was unbalanced, as expected the model obtained better results for the highest frequency class, therefore, it was decided to use the precision vs recall curve metric to optimally measure each class. As a next approach, the classes can be balanced using the following technique: **Smote**

In order to show the results in an interactive way and make it easier for the user to test the model, a website was created using "Python" programming language with "Tornado" framework. The user can write any text and the web returns the result of the classification. 
