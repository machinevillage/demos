from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import time
import matplotlib
import io
import yelper

app = Flask(__name__)


@app.route('/')
def hello():
    return (render_template("index.html"))


@app.route('/response', methods=['POST', 'GET'])
def response():

    rest = request.form.get("rest")
    loc_ = request.form.get("loc")

    print(rest)
    print(loc_)

    global new
    new = yelper.main(rest, loc_)
    new = new[['rev', 'score', 'rating', 'pos', 'neg']]
    new.columns = ['Review', 'Sentiment Score', 'Star Rating', 'Positive Words', 'Negative Words']
    new.reset_index(drop=True, inplace=True)

    return (render_template("index.html", rest = rest, loc=loc_, revs=new, tables=[new.to_html(classes='data')], titles=new.columns.values))

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)
