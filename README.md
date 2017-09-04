# Style Transfer using SqueezeNet #

Renders a given content image in the style of another given image through the power of deep learning and CNNs. The architecture is modeled after the one presented in Gatys, L.A., Ecker, A.S., Bethge, M.: A neural algorithm of artistic style. [arXiv:1508.06576] (2015)

### Requirement ###

- Python >=3.4
  - TensorFlow >=1.0
- Node >=6.9


### How to run ###

    $ pip install -r requirements.txt
    $ npm install
    $ gunicorn main:app -t 600 --log-file=-


### Deploy to Heroku ###

    $ heroku apps:create [NAME]
    $ heroku buildpacks:add heroku/nodejs
    $ heroku buildpacks:add heroku/python
    $ git push heroku master

or Heroku Button.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)
