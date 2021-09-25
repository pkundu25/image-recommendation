import os
import numpy as np
import pandas as pd
from PIL import Image
import glob
import cv2
from tensorflow.keras.models import load_model
from scipy.spatial import distance
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.backend.clear_session() 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')



print("Running")
dataset_path = ".\\imgset\\"

categories = {
    0 : "single-t-shirt",
    1 : "double-t-shirt",
    2 : "multiple-t-shirt"
}

Category_Id = dict((g, i) for i, g in categories.items())



def images_path(types,dataset_path):
  cat_path=[]
  for img in glob.glob(dataset_path + str(types) + "/*.jpeg"):  #reading all the images using glob functionality
    dum = [img,types]
    #print(img)
    cat_path.append(dum)  #adding image along with category 
  print(types + ": " +str(len(cat_path)))

  return cat_path


all_images =[]

single_t_shirt = images_path("single-t-shirt", dataset_path)
all_images.append(single_t_shirt)

double_t_shirt = images_path("double-t-shirt", dataset_path)
all_images.append(double_t_shirt)

multiple_t_shirt = images_path("multiple-t-shirt", dataset_path)
all_images.append(multiple_t_shirt)


imgs_path =[]
cat =[]
for types in all_images:
  for img in types:
    imgs_path.append(img[0])
    cat.append(img[1])
len(cat), len(imgs_path)


def Img_preprocesser_aen(dataset):
  images =[]
  for image in dataset['Image']:
    img = cv2.imread(image)      #reading image using opencv
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)  #resizing the image "norm_image = interval_mapping(image, 0, 255, 0.0, 1.0)""
    img = np.array(img) / 255.0 #normalizing the image
    images.append(img)
  return images


dataset = pd.DataFrame(columns=['Image','Category'])
#print(imgs_path)
dataset['Image'] = imgs_path
dataset['Category'] = cat
print(dataset.tail())


def loadingmodel():
  print("Hi----------------------------------------------")
  txt = "CNN_autoencoder_latest_model_mse.h5" 
  x = txt.encode()
  model = load_model(txt)
  print(model.summary())
  return model

#model = loadingmodel()
def redefiningmodel(model):
  encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
  dataset_images = Img_preprocesser_aen(dataset)
  dataset_images = np.reshape(dataset_images,(len(dataset_images),128,128,3))

  return encoder

def making_recommondataions(inputvariable, dataset):
  from scipy.spatial import distance
  inputvariable = np.reshape(inputvariable,(-1,1))
  cosine_scores={}
  count=0
  for i in dataset:
    x = np.reshape(i,(-1,1))
    score = distance.cosine(inputvariable,x)
    cosine_scores[count] = score
    count += 1
  print(len(cosine_scores))
  sorted_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=False)
  top10 = sorted_scores[:6]
  indexes =[]
  for i in top10:
    indexes.append((i[0],(1-i[1])*100))

  return indexes
  
  
#input_encoder = redefiningmodel(input_model)
def predictencoders(encoder, dataset_images, rand):
  pred_encoder = encoder.predict(dataset_images)
  res_indexes = making_recommondataions(pred_encoder[rand],pred_encoder)
  images=[]
  for ind in res_indexes:
    temp={}
    img = Image.open(dataset.iloc[ind[0]]['Image'])
    cat = dataset.iloc[ind[0]]['Category']
    level = dataset.iloc[ind[0]]['Image'].split('/')[-1]
    score = ind[1]
    temp["type"] = cat
    temp["filename"] = level
    temp['matching'] = score
    images.append(temp)  #[cat,level,score]

  return images


def getsimilar(imgname):
  print("Hi-------------------------")
  #dataset.to_csv("abc.csv")	
  #for im,i in zip(dataset['Image'], dataset.shape[0]):
  for im,i in zip(dataset['Image'], range(dataset.shape[0])):
    name = str(im.split('/')[-1])
    print(imgname, im, name, i)
    if(imgname==name[:-4]):
      ind = i
      break

  processed = Img_preprocesser_aen(dataset)
  model = loadingmodel()
  encoder = redefiningmodel()
  result = predictencoders(encoder, processed,ind)


  return result



"""
result = [cat,level,score]
//example [fibers,filename,%matching]
"""