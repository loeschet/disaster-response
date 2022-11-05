import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), "./models"))
from train_classifier import ColumnSelector, tokenize
from app.run import run_app

if __name__ == "__main__":
    run_app(debug=False)
