from flask import Flask, request, render_template
import os
import pandas as pd
from application.house_predict import load_models, get_prediction

app = Flask(__name__)
data_path = os.path.join(app.root_path, 'models')
load_models(data_path)

@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    data_dict = {}

    try:
        for item in data:
            data_dict[item]=[data[item]]

        features = pd.DataFrame(data_dict)
        for feature in ['sqft', 'zipcode', 'beds', 'stories', 'private_pool', 'lotsize', 'baths_count', 'year_built']:
            features[feature] = features[feature].astype('int32')

        result = get_prediction(features)

        return render_template('form.html',
                           street=data['street'],
                           sqft=data['sqft'],
                           zipcode=data['zipcode'],
                           beds=data['beds'],
                           stories=data['stories'],
                           private_pool=data['private_pool'],
                           lotsize=data['lotsize'],
                           baths_count=data['baths_count'],
                           year_built=data['year_built'],
                           result=result
                           )
    except Exception:
        return render_template('form.html',
                               street=data['street'],
                               sqft=data['sqft'],
                               zipcode=data['zipcode'],
                               beds=data['beds'],
                               stories=data['stories'],
                               private_pool=data['private_pool'],
                               lotsize=data['lotsize'],
                               baths_count=data['baths_count'],
                               year_built=data['year_built'],
                               result='нет_оценки'
                               )

if __name__ == '__main__':
    app.run('localhost', 5000)