# Style Transfer using SqueezeNet #

[See it in action!](https://style-transferer.herokuapp.com/)

Renders a given content image in the style of another given image through the power of deep learning and CNNs. The architecture is modeled after the one presented in Gatys, L.A., Ecker, A.S., Bethge, M.: A neural algorithm of artistic style. [arXiv:1508.06576] (2015)

+ SqueezeNet is a CNN that we use as a feature extractor. It gives AlexNet-level performance with 50x fewer parameters and <0.5MB model size.


### Requirement ###

- Python >=3.4
  - TensorFlow >=1.0
- Node >=6.9


### How to run ###

    $ pip install -r requirements.txt (or pip3 to use the python3 distribution)
    $ npm install
    $ gunicorn main:app -t 6000 --log-file=-
