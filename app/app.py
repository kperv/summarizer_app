#!/usr/bin/env python3
from flask import Flask, render_template, request, url_for


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        input_text = request.form['text']
        input_number = request.form['number']
        text = [input_text]
        number = int(input_number)
        result = number
    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)