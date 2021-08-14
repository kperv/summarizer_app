from flask import Flask, render_template, request, url_for


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request_method == 'POST':
        form = request.form
        text = form['text']
        number = form['number']
        result = [text, number, text]
        return render_template("index.html", result=result)
    render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)