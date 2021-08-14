#!/usr/bin/env python3
from flask import Flask, render_template, request, url_for


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form = request.form
        text = form['text']
        number = form['number']
        result = [text, number, text]
        return render_template("index.html", result=result)
    render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)