"""
    ResNet50 Finetuning
    Finetuning ResNet50 using
    ILSVRC pre-trained model.
    The dataset consits 113205 images of
    1000 plant species, Dataset from PlantClef
    2015 Competition on ImageClef.org.
    
    Implementation using Keras high level
    library running on tensorflow
    backend.

    Data is resized and classes id saved in
    img_classes.npy where [image id, label].
    
    Evaluation used: *categorical_cross_entropy,
    using cross-validation method.
    
    @author: rabiaabuaqel
    """


################################################
#IMPORTS
################################################
from keras.applications.resnet50 import ResNet50
from sklearn.cross_validation import train_test_split
import os
import numpy as np
from time import time
from PIL import Image
from keras import metrics
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math



################################################
#VARS
################################################
DATA_DIR = '/data'
SAMPLES = 113204
CLASSES = 1000
SIZE = (224, 224)
BATCH_SIZE = 16
X = []
Y = []




################################################
#LOAD DATA
################################################
# load data and labels through iterating
# the saved numpy array
ids_and_labels = np.load("imgs_classes.npy")
for id in ids_and_labels:
    X.append(np.asarray(Image.open(DATA_DIR+TRAIN_DIR+'/'+str(id[0])+'.jpg')))
    Y.append(id[1])
# convert data and labels array to numpy arrays
X = np.array(X)
Y = np.array(Y)



################################################
#SPLIT TRAIN-VAL-TEST
################################################
# using sklearn.cross_validation
X_train, X_val, y_train, y_val = \
train_test_split(X, Y, test_size=0.3, random_state=1)



################################################
#LOAD MODEL
################################################
# load model with ILSVRC weights
model = ResNet50(weights='imagenet')
# remove output layer of the model
model.layers.pop()
# freeze all layers except last 12 layers
for layer in model.layers[:15]:
    layer.trainable=False
# add dropout
last = model.layers[-1].output
x = Dropout(0.5)(last)
# add Softmax regression for output layer
# and attatch model to new Dense layer
last = model.layers[-1].output
x = Dense(CLASSES, activation="softmax")(last)
finetuned_model = Model(model.input, x)

def top_5_accuracy(y_true, y_pred):
    """
        Customize keras metrics
        func
        """
    return metrics.sparse_top_k_categorical_accuracy(y_true,
                                                     y_pred, k=5)

def top_3_accuracy(y_true, y_pred):
    """
        Customize keras metrics
        func
        """
    return metrics.sparse_top_k_categorical_accuracy(y_true,
                                                     y_pred, k=3)
# compile model
finetuned_model.compile(optimizer=Adam(lr=0.0001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy',top_5_accuracy, top_3_accuracy])
# outputs
finetuned_model.classes = y_train
# ealry stopping if val_acc doesnt improve
# in 5 epochs
early_stopping = EarlyStopping(monitor='val_acc', patience=10,
                               verbose=2, mode='auto')
# save best weights after each epoch
checkpointer = ModelCheckpoint('resnet50_best_planetclef.h5',
                               verbose=2, save_best_only=True)



################################################
#TRAINING
################################################
t0 = time()
# fit data to model
history = finetuned_model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                              epochs=50, verbose=2,
                              callbacks=[early_stopping, checkpointer],
                              validation_split=0.0,
                              validation_data=(X_val,y_val), shuffle=True,
                              class_weight=None, sample_weight=None,
                              initial_epoch=0)
print("Model Training Time: %0.3fs" % (time() - t0))
# save final weights
finetuned_model.save('resnet50_final_planetclef.h5')


################################################
#EVALUATION
################################################
X_test = []
Y_test = []
ids_and_labels = np.load("test_imgs_classes.npy")
for id in ids_and_labels:
    X_test.append(np.asarray(Image.open("/data2"+"/test/"+str(id[0])+'.jpg')))
    Y_test.append(id[1])
np.set_printoptions(threshold=np.inf)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


t0 = time()
score = finetuned_model.evaluate(X_test, Y_test,
                       batch_size=BATCH_SIZE, verbose=1)
print("Model Evaluation Time: %0.3fs" % (time() - t0))

print ("Model Evaluation Score:")
print (score)
################################################



