from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
import pickle

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from werkzeug.local import ContextVar
from werkzeug.serving import run_simple
from gevent.pywsgi import WSGIServer

from run import getsimilar

  


IMAGES_PATH = './imgset/'
CATEGORIES = ['single-t-shirt','double-t-shirt','multiple-t-shirt']
'''
def get_image_details():
	img_list=[]
	for cat in CATEGORIES:
		for im in glob.glob(IMAGES_PATH+cat+'/*.jpeg'):
			name = im.split("/")[-1]
			img_list.append(name.split('\\')[1])

	details={
	"images": img_list,
	"types": CATEGORIES
	}
	return details

def get_cat_images():
	cat_images={}
	for cat in CATEGORIES:
		imgs=[]
		for im in glob.glob(IMAGES_PATH+cat+'/*.jpeg'):
			name = im.split("/")[-1]
			imgs.append(name.split('\\')[1])
		cat_images[cat]=imgs
	return [cat_images['single-t-shirt'],cat_images['double-t-shirt'],cat_images['multiple-t-shirt']]
'''


app = Flask(__name__)
print('Application running on http://127.0.0.1:5000/(localhost:5000)')

'''
@app.route('/', methods=['GET'])
def index():
	details = get_image_details()
	return render_template('base.html',data=details)
'''	


@app.route('/')
def index():
    return render_template('index(1).html')	
	

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    print(uploaded_file.filename)
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        result = getsimilar(uploaded_file.filename)
    return redirect(url_for('index(1)'))
	
@app.route('/')
def home():
    return render_template('demo-html.html')		

@app.route('/similarimages', methods=['GET', 'POST'])
def getsimilarimages():
    if request.method == 'POST':
    	data = request.data.decode()
    	result = getsimilar(data)
    	return result
        
    return ''

'''
@app.route('/images', methods=['GET'])
def getimagedetails():
	details = get_cat_images()
	return render_template('images.html',data={'single-t-shirt':details[0],'double-t-shirt':details[1],
												'multiple-t-shirt':details[2]
												})
'''												
												
if __name__ == '__main__':
    app.run(debug=True)