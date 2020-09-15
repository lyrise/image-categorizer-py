import datetime
import glob
import os
import random
import shutil
import sys

import keras.preprocessing.image as image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

src_dir = sys.argv[1]
dest_dir = sys.argv[2]

np.random.seed(0)

img_paths = []
for ext in ('*.gif', '*.png', '*.jpg'):
    for file in glob.glob(os.path.join(src_dir, "**/"+ext), recursive=True):
        img_paths.append(file)
random.shuffle(img_paths)
img_paths = img_paths[:10000]

img_list = []
failed_img_paths = []
for p in img_paths:
    try:
        img = image.load_img(p, target_size=(224, 224), grayscale=False)
        x = image.img_to_array(img)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        img_list.append(x)
    except OSError:
        failed_img_paths.append(p)
img_list = np.array(img_list)

for p in failed_img_paths:
    img_list.remove(p)

model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, input_shape=[224, 224, 3], weights='imagenet')

features = model.predict(img_list)
dataset = features.reshape((len(img_list), -1))

cluster_count = 50

kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(dataset)
labels = kmeans.labels_

datetime_text = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

for i in range(cluster_count):
    label = np.where(labels == i)[0]
    target_dir = os.path.join(dest_dir, datetime_text, str(i))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for j in label:
        org_path = img_paths[j]
        new_path = os.path.join(target_dir, os.path.basename(org_path))
        shutil.move(org_path, new_path)
