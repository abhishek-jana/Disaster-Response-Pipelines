import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load files in pandas dataframe

    INPUT
    messages_filepath - PATH to disaster_messages file
    categories_filepath - PATH to disaster_categories file

    OUTPUT
    Dataframe - returns pandas dataframe of messeges and categories merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on = 'id')
    return df


def clean_data(df):
    """
    Function to clean database

    INPUT
    Dataframe - Pandas dataframe to be cleaned

    OUTPUT
    Dataframe - Cleaned pandas dataframe
    """
    # Split categories into separate category columns
    categories = df["categories"].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace categories column in df with new category columns.
    df = df.drop(['categories'],axis=1) # drop the original categories column
    df = pd.concat([df,categories],join = 'inner',axis=1) # concatenate
    # drop the child-alone column from `df`
    df = df.drop(['child_alone'],axis=1)
    # drop the rows from related column with value = 2
    df=df[df["related"] < 2]
    df = df.drop_duplicates() # drop duplicates
    return df


def save_data(df, database_filename):
    """
    Funtion to save df into sqlite3

    INPUT
    dataframe - Pandas Dataframe
    string - path ot the database filename

    OUTPUT
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql( 'DisasterResponseTable' , engine,if_exists='replace', index=False)


def main():
    """
    Function to load, clean and save data in sqlite3 database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print(f'Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
