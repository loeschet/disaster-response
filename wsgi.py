import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), "./models"))
from train_classifier import ColumnSelector, tokenize
from app.run import init_app

app = init_app()

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
