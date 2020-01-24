# classifier/application/routes.py
from flask import Flask, request, jsonify
from application import app
from application.spam_classifier import classify

#@app.route('/hello_user', methods=['POST'])
#@app.route('/increase_number', methods=['POST'])
@app.route('/classify_text', methods=['POST'])

#def hello_user():
#    data = request.json
#    user = data['user']
#    return f'hello {user}'

#def increase_number():
#    data = request.json
#    number = data['number']
#    return str(number+1)

def classify_text():
    data = request.json
    text = data.get('text')
    if text is None:
        params = ','.join(data.keys())
        return jsonify({'message': f'Parameter "{params}" is invalid'}), 400
    else:
        result = classify(text)
        return jsonify({'result': result})