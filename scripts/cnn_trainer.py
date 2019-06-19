import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
import argparse

import gestures_recorder as gr

import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from sklearn.model_selection import train_test_split

# how many images to process before applying gradient correction
batch_sz = 32

# how many times the network should train on the whole dataset
nb_epoch = 200

# how many images to generate per image in datasets
nb_gen = 20

# create path if not exists
def create_ifnex(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print "create floder: "+directory

# exit program is path if not exists
def exit_ifnex(directory):
    if not os.path.exists(directory):
        print(directory, 'does not exist')
        exit()

# loads an opencv image from a filepath
def get_img(path):
    image = cv2.imread(path, 0) if gr.grayscale else cv2.imread(path, gr.channel)
    image = cv2.resize(image, (gr.width, gr.height))
    image = img_to_array(image)
    image = image.reshape(gr.width, gr.height, gr.channel)
    return image

# use keras to generate more data from existing images
def generate_data(path):
    if nb_gen==0:
        return
	datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
	classesFolders = os.listdir(path)
	for folder in classesFolders:
		files = os.listdir(os.path.join(path, folder))
		for fl in files:
			img = get_img(os.path.join(path, folder, fl))
			img = img.reshape(1, gr.width, gr.height, gr.channel)
			i = 0
			for batch in datagen.flow(img, batch_size=1, save_to_dir=os.path.join(path, folder), save_prefix='genfile', save_format=gr.file_format):
				i += 1
				if i > nb_gen:
					break

def load_data(dataset_path):
    x_data = []
    y_data = []
    labels = []

    classes = os.listdir(dataset_path)
    for i in range(len(classes)):
        files = os.listdir(os.path.join(dataset_path, classes[i]))
        labels.append(classes[i])
        for fl in files:
            x_data.append(get_img(os.path.join(dataset_path, classes[i], fl)))
            y_data.append(i)

    x_data = np.array(x_data, dtype="float") / 255.0
    y_data = np.array(y_data)

    y_data = keras.utils.np_utils.to_categorical(y_data)
    return x_data, y_data, labels

# split dataset into training and testing
def split_dataset(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
    return x_train, y_train, x_test, y_test

# build convolutional neural network
def build_model(nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu', input_shape=[gr.height, gr.width, gr.channel], kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()
    plot_model(model, to_file='../model/modified_letnet_model.png', show_shapes=True)
    return model

# train model with data
def train(model, x_train, y_train):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_sz, epochs=nb_epoch, verbose=1, validation_split=0.3)
    return history

# save network model and network weights into files
def save_model(model, network_path):
    create_ifnex(network_path)
    open(os.path.join(network_path, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, 'weights.h5'), overwrite=True)

# load network model and network weights from files
def read_model(network_path):
    exit_ifnex(network_path)
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model

def plot_history(history):
    #  Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def main():
    # parse arguments
    global nb_epoch,batch_sz,nb_gen
    parse=argparse.ArgumentParser()
    parse.add_argument('-epoch',nargs='?',type=int,default=nb_epoch,help="number of epochs, default: "+str(nb_epoch))
    parse.add_argument('-batch',nargs='?',type=int,default=batch_sz,help="batch size, default: "+str(batch_sz))
    parse.add_argument('-gen',nargs='?',type=int,default=nb_gen,help="number of images to be generated, default: "+str(nb_gen))
    args=vars(parse.parse_args())
    nb_epoch=args['epoch']
    batch_sz=args['batch']
    nb_gen=args['gen']
    # display information
    print ('#'*50)
    print "batch size:"+str(batch_sz)
    print "number of epochs: "+str(nb_epoch)
    print "generated data per image: "+str(nb_gen)
    print ('#'*50)
    # check path
    create_ifnex('../model')
    # generate data
    generate_data(gr.dataset_folder)

    # Load data, split data
    x_data, y_data, labels = load_data(gr.dataset_folder)
    x_train, y_train, x_test, y_test = split_dataset(x_data, y_data)

    # Create network, train it, save it
    nb_classes = len(os.listdir(gr.dataset_folder))
    model = build_model(nb_classes)
    history = train(model, x_train, y_train)
    save_model(model, '../model')

    # Write labels
    fl = open('../model/labels.txt', 'w')
    for item in labels:
        fl.write("%s\n" % item)

    # Evaluate model on test data
    scores = model.evaluate(x_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # display graphs
    plot_history(history)

if __name__ == '__main__':
    main()
