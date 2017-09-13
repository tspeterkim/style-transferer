import os
import requests
from flask import Flask, jsonify, render_template, request

# from styletransfer import sq model
from styletransfer.squeezenet import SqueezeNet
from styletransfer.style import style_transfer, get_session

# redis
from rq import Queue
from worker import conn

def return_none(str):
    return None

q = Queue(connection=conn) # create redis queue
j = None
# sess = get_session()
# model = SqueezeNet(sess=sess)

# init webapp
app = Flask(__name__)

@app.route('/api/styletransfer', methods=['POST'])
def styletransfer():
    print (request.json)
    global j
    j = q.enqueue(style_transfer, **request.json)
    return jsonify(result='transfer initiated...')

@app.route('/api/checkTransferStatus', methods=['GET'])
def checkTransferStatus():
    if j != None and j.result != None:
        print ("Transfer complete!")
        return jsonify(result=j.result)
    return jsonify(result=None)

@app.route('/api/download_img', methods=['POST'])
def download_img():
    url = request.json['url']
    name = url.split('/')[-1]
    r = requests.get(url)

    with app.open_instance_resource(name, 'wb') as f:
        f.write(r.content)

    return jsonify(img=name);

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
