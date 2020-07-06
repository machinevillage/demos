from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import time
import twit
import matplotlib
import io

app = Flask(__name__)

def plotit(dates, senti):

    df = pd.DataFrame({'dates':dates, 's':senti})
    df.index=dates
    df.drop('dates', 1,inplace=True)

    plt=df[["s"]].resample("3d").mean().plot(figsize=(15,4))
    bytes_image = io.BytesIO()
    plt.figure.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return (bytes_image)

def plotit2(dates, senti):

    df = pd.DataFrame({'dates':dates, 's':senti})
    df.index=dates
    df.drop('dates', 1,inplace=True)

    plt=df[["s"]].resample("3d").mean().plot(figsize=(15,4))
    fig = df.plot(figsize=(10,5), linewidth=5, fontsize=20)
    
    global new_graph_name2
    new_graph_name2 = "graph" + str(time.time()) + ".png"

    fig.figure.savefig('static/' + new_graph_name2)

def plotit3(dates, senti, tweets):
    
    df = pd.DataFrame({'dates':dates, 'sentiment':senti, 'tweet':tweets})
    df.rename(columns={'dates':'date', 'sentiment':'sentiment_score'}, inplace=True)

    print(df.head())

    fig = px.line(df, x="date", y="sentiment_score", 
                 hover_name="tweet")
    #, hover_data=["tweet"])
    
    global new_graph_name3
    new_graph_name3 = "graph" + str(time.time()) + ".html"

    plot(fig, filename='static/' + new_graph_name3)


@app.route('/')
def hello():
    return (render_template("index.html"))

#@app.route('/plot_new2.png', methods=['GET'])
def plot_png():

    print('inside')
    print(new[2][:5])
    print(new[0][1])
    bytes_obj = plotit3(new[1], new[2], new[0])
    #global new_graph_name
    #new_graph_name = "graph" + str(time.time()) + ".png"
    
    #print(new[1][:5])
    #send_file(bytes_obj,
    #                 attachment_filename='static/' +  new_graph_name,
    #                 mimetype='image/png')


@app.route('/response', methods=['POST', 'GET'])
def response():

    hand = request.form.get("hand")
    print(hand)

    global new
    new = twit.main(hand)

    #print(new[1][:5])
    print('first')
    print(new[2][:5])
    print(new[0][1])

    #bytes_obj = plotit(new[1], new[2])
    plot_png()

    #send_file(bytes_obj,
    #                 attachment_filename='plot_new.png',
    #                 mimetype='image/png')
    return (render_template("index.html", hand = new, graph=new_graph_name3))

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000)
