# Disaster Response Pipeline Project

In this project, we'll apply ETL pipeline and ML pipeline to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
### Project Set Up and Installation
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project motivation
This is one of the most important problem in data science and machine learining. During a disaster, we get millions and millions of messeges either direct or via social media. We'll probably see 1 in 1000 messeges that are relevent. Few importatnt words like water, blocked road, medical supplies are used during a disaster response. We have a categorical dataset with which we can train an ML model to see if we identify which messeges are relevant to disaster response.
 

### Dataset


### Project structure



### Description of the repository.




### Results



### Future work

 

### Acknowledgements

I am thankful to Udacity Data Science Nanodegree program and figure eight for motivating me in this project.

I am also thankful to figure eight for making the data publicly avilable.



