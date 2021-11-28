import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request, jsonify
import json
import requests
from sqlalchemy import *


app = Flask(__name__)

ds = pd.read_csv("./sample-data.csv")
results = {}

def execute(desc):
    global ds
    ds = ds.append({"id":1000,"description":desc,"action":"none"},ignore_index=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['description'])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['id'][i], ds['action'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]
        
    


def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(desc, num):
    global results
    global ds
    print("Recommending " + str(num) + " items similar to " + desc + "...")
    print("-------")
    execute(desc)
    recs = results[1000][:num]
    output = []
    for rec in recs:
        print("Action: "+str(rec[2])+" | Recommended: " + item(rec[1]) + " (score:" + str(rec[0]))
        output.append("Action: "+str(rec[2])+" | Recommended: " + item(rec[1]) + " (score:" + str(rec[0]))
    ds = pd.read_csv("./sample-data.csv")
    results = {}
    return output

@app.route("/", methods=['GET'])
def home():
    query = request.args.get('query')
    return jsonify(recommend(query, num=3))

if __name__ == "__main__":
    app.run()
 