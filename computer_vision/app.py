from flask import Flask, request, render_template

import model



app = Flask(__name__)



@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/response', methods=['POST'])
def response():

    url = request.form.get("url")
    print(url)

    new = model.run_model(url)
    return render_template("index.html", note = new, url=url)

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)

