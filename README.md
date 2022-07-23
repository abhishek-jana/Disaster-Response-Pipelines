# Disaster Response Pipeline Project

In this project, we'll apply ETL pipeline, NLP pipeline, and ML pipeline to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

**Main Web Interface**
![Main web page](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/interface.png)

### Project Motivation
This is one of the most important problem in data science and machine learning. During a disaster, we get millions and millions of messages either direct or via social media. We'll probably see 1 in 1000 messages that are relevant. Few important words like water, blocked road, medical supplies are used during a disaster response. We have a categorical dataset with which we can train an ML model to see if we identify which messages are relevant to disaster response.

In this project three main features of a data science project have been utilized:

1. **Data Engineering** - In this section I worked on how to Extract, Transform and Load the data. After that I prepared the data for model training. For preparation I cleaned the data by removing bad data (**ETL pipeline**) then used NLTK to tokenize, lemmatize the data (**NLP Pipeline**). Finally used custom features like StartingVerbExtractor, StartingNounExtractor to add new to the main dataset.
2. **Model Training** - For model training I used [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html) to create the **ML pipeline**.
3. **Model Deployment** - For model deployment, I used the flask API.

### Project Set Up and Installation

This project is done on anaconda platform using jupyter notebook jupyter notebook. The detailed instruction of how to install anaconda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
To create a virtual environment see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

in the virtual environment, clone the repository :
```
git clone https://github.com/abhishek-jana/Disaster-Response-Pipelines.git
```
Python Packages used for this project are:
```
Numpy
Pandas
Scikit-learn
xgboost
NLTK
regex
sqlalchemy
Flask
Plotly
```
To install the packages, run the following command:

`pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project structure

The project is structured as follows:

![structure](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/dir_structure.png)

### Description of the repository.

**data** folder contains the data "disaster_categories.csv", "disaster_messages.csv" to extract the messages and categories.
"DisasterResponse.db" is a cleaned version of the dataset save in sqllite database.
"ETL Pipeline Preparation.ipynb" is the jupyter notebook explaining the data preparation method.
"process_data.py" is the python script of the notebook.

"ML Pipeline Preparation.ipynb" is the jupyter notebook explaining the model training method. The relevant python file "train_classifier.py" can be found in the **models** folder.
Final trained model is saved as "classifier.pkl" in the **models** folder.

**app** folder contains the "run.py" script to render the visualization and results in the web. **templates** folder contains the .html files for the web interface.

### Results and Evaluation
The accuracy, precision and recall are:

**accuracy**


![acc](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/accuracy.png)

**precision and recall**

![pre](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/precision.png)

Some of the predictions on messages are given as well:

**message 1**

![m1](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/message1.png)

**message 2**

![m2](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/message2.png)

**message 3**

![m3](https://github.com/abhishek-jana/Disaster-Response-Pipelines/blob/main/images/message3.png)

### Future work

In future I am planning to work on the following areas of the project:

1. Testing different estimators and adding new features in the data to improve the model accuracy.

2. Add more visualizations to understand the data.

3. Improve the web interface.

4. Based on the categories that the ML algorithm classifies text into, advise some organizations to connect to.

5. This dataset is imbalanced (ie some labels like water have few examples). In the README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.

### Acknowledgements

I am thankful to Udacity Data Science Nanodegree program and figure eight for motivating me in this project.

I am also thankful to figure eight for making the data publicly available.
