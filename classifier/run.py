# classifier/run.py
from application import app
from application.spam_classifier import train
import os

data_path = os.path.join(app.root_path, 'spam_or_not_spam.csv')
get_data(data_path)
train()
