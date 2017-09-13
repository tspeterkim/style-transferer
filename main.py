import os
import requests
from flask import Flask, jsonify, render_template, request

# from styletransfer import sq model
from styletransfer.squeezenet import SqueezeNet
from styletransfer.style import style_transfer, get_session

# redis
from rq import Queue
from worker import conn

q = Queue(connection=conn) # create redis queue

# sess = get_session()
# model = SqueezeNet(sess=sess)

# init webapp
app = Flask(__name__)

@app.route('/api/styletransfer', methods=['POST'])
def styletransfer():
    # request.json['sess'] = sess
    # request.json['model'] = model
    print (request.json)
    # r = style_transfer(**request.json)

    job = q.enqueue(style_transfer, **request.json)
    while (job.result == None):
        x = True
    print ("done!")
    return jsonify(result=job.result)

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
