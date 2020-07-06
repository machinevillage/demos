from flask import Flask, request, render_template
from textblob import TextBlob

app = Flask(__name__)


def get_sentiment(in_str):

    sent = TextBlob(in_str).sentiment.polarity
    if sent>.5:
        sentiment='great'

    elif sent>=0:
        sentiment='neutral'
    elif sent>-.5:
        sentiment='bad'
    else:
        sentiment='terrible'

    return (sentiment)


@app.route('/')
def response():

    input_text = request.args['text']
    note = get_sentiment(input_text)
    return note 

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)
