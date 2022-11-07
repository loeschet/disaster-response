# disaster-response
Disaster response pipeline for twitter posts.

This is a web app using a machine learning model to predict disaster categories (e.g. floods, fires, medical emergencies etc.) from input tweets.

The models used in this app are `scikit-learn` classifier models in combination with natural language preprocessing based on `nltk`. The data used here was provided by [appen](https://appen.com) (formally figure eight) and contains several tweets related to different disaster types for training.

The app can either be run locally, but there is also the option to use it via web at [https://disaster-response.particlephysics.de](https://disaster-response.particlephysics.de).

## Recommended software

To be able to run this project, several `python` packages are required. A full list is contained in the `requirements.txt` file. However, I added a short list of the packages here as well for convenience:

- `dash`
- `plotly`
- `Flask`
- `nltk`
- `numpy`
- `pandas`
- `scikit-learn`
- `SQLAlchemy`
- `tables`

Other required software for the above packages is installed automatically when using `pip`.

## Running locally

### 1. Run ETL pipeline

The ETL (Extract, Transform, Load) pipeline is the first step to run. After cloning the repository, `cd` into its directory and run the following command:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

This step will create a `pandas` dataframe from the message and categories data, clean it up and store it into a `SQL` database under the `data` directory.

### 2. Run machine learning pipeline

The training of the `scikit-learn` model can be run using the following command:

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will load the data from the database created in step 1. and train a `scikit-learn` model which is then saved as a pickle file under the `models` directory.

**Important Note:** During the training, grid search with cross-validation is performed to find the optimal parameters of the model. This was tested on a modern CPU and found to take approximately 45 minutes. In case you don't want to run the full grid search and just do a quick test of this project, please open up `models/train_classifier.py` and comment out lines 175-196, which will cause the program to simply use the default settings of the model in `scikit-learn`.

### 3. Run web app

Finally, you can simply run the web app using the following command:

```
python app/run.py
```

This will start a `Flask` app showing the results of your trained model. You can access it by typing `127.0.0.1:3001` in your browser window.

## (Optional) Deploy to website

This step is completely optional and the technical details reach far beyond this introductory README file. However, note that I included several files that allow you to deploy this app to a server using `dokku`. These files are:
- `Procfile`
- `nltk.txt`
- `requirements.txt`
- `runtime.txt`
- `wsgi.py`

For more information about how `dokku` works and how to deploy apps to your server, check the [dokku documentation](https://dokku.com/docs/getting-started/installation/). For a live demo how the result looks on my own VPS, check out this URL: https://disaster-response.particlephysics.de/

## How the app works

### Tweet classification

In the upper part of the app, there is a bar where you can put in any text (preferrably, a tweet) which gets classified in one or multiple of the 36 different disaster categories by the deployed model based on its content.

Once you put in a text and click on the "Classify Message" button, you will be presented a list with all of the categories. If the model thinks your input text fits in any of the disaster categories, these will be highlighted in green. You can repeat this process multiple times and with any text. Feel free to play around with it, copy/paste some tweets from twitter or make up your own :-)

### Exploratory data analysis

To get some feeling about how the dataset looks like, I included some plots that might be interesting, using `plotly` and an interactive `dash` app.

The first plot simply shows the distribution of message "genres", which are *direct*, *news* and *social*.

Below this plot, there are two interactive plots that both react to the drop-down list: From this list, you can choose any of the available disaster categories. The center plot will then show the 5 most occurring words (or, more precisely, "tokens", which the model trains on) for that category. The bottom plot shows for tweets in the chosen category, how many tweets are also flagged positive for any of the other categories (i.e. it shows the "correlation" of the chosen disaster type with the other categories).