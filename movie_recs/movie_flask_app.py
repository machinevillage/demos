from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import movie_model

app = Flask(__name__)


@app.route('/')
def response():

    input_text = request.args['input_movies']
    movie_preds = movie_model.main(input_text.split(','))
    return movie_preds

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)

