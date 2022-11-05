import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - path to csv file containing tweeted messages about
                        disasters
    categories_filepath - path to csv file containin disaster categories
    
    OUTPUT:
    df - pandas dataframe merged from categories and messages csv file
    
    Function to read in twitter messages and respective categories from two
    provided csv files and merge them into a single pandas dataframe.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    '''
    INPUT:
    df - pandas dataframe to be cleaned
    
    OUTPUT:
    df - cleaned pandas dataframe
    
    Function to clean up an input pandas dataframe. Creates one-hot encoded
    label columns for each of the tweet disaster categories, merges them into
    the original dataframe and drops duplicate information.
    '''
    # create integer-valued category columns that indicate which categories
    # the twitter post belongs to
    categories = df['categories'].str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # extract label (0 or 1) from strings and convert to integer
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # drop original categires columns and concatenate with categories DataFrame
    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates and double-check if any duplicates still exist
    df.drop_duplicates(inplace=True)
    print(f"Number of duplicates: {df.duplicated().sum()}")
    
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - pandas dataframe to be stored in SQL databes file
    database_filename - filename where database should be stored
    
    This function stores and input pandas dataframe in a SQLite database.
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DisasterResponse", engine, index=False, if_exists="replace")


def main():
    
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
         database_filepath) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()