from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = {}
    for item in data:
        features[item]=data[item]

    print(features)
    return render_template('form.html',
                           #city=data['city'],
                           sqft=data['sqft'],
                           zipcode=data['zipcode'],
                           beds=data['beds'],
                           #state=data['state'],
                           stories=data['stories'],
                           private_pool=data['private_pool'],
                           lotsize=data['lotsize'],
                           baths_count=data['baths_count'],
                           year_built=data['year_built'],
                           remodeled_year=data['remodeled_year'],
                           result='test'
                           )


if __name__ == '__main__':
    app.run('localhost', 5000)
