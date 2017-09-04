import os
import requests
from flask import Flask, jsonify, render_template, request

# from mnist import model
from styletransfer.squeezenet import SqueezeNet
from styletransfer.style import style_transfer, get_session

sess = get_session()
model = SqueezeNet(sess=sess)

# webapp
app = Flask(__name__)

@app.route('/api/styletransfer', methods=['POST'])
def styletransfer():
    print (request.json)
    r = style_transfer(sess, model, **request.json)
    return jsonify(result=r)

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
