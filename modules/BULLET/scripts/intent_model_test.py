import pandas as pd
import numpy as np
import keras
import tensorflow
import h5py
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import sgd
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('../bin/resources/data/0103.csv', delim_whitespace = True)

intent = data.values[0:len(data):200,35]
gripper_pos = data.values[0:len(data):200,36:39]
gripper_rot = data.values[0:len(data):200,39:48]
gripper_angle = data.values[0:len(data):200,48]

args = np.argwhere(intent != 0)
intent_relabel = intent[args]
gripper_pos_relabel = gripper_pos[args[:,0], :]
gripper_rot_relabel = gripper_rot[args[:,0], :]
gripper_angle_relabel = gripper_angle[args]
print intent_relabel.shape

intent_classes  = np_utils.to_categorical(intent_relabel-1, 4)

input_matrix = np.hstack([gripper_pos_relabel, gripper_rot_relabel, gripper_angle_relabel])

sequence_length = 3
input = np.empty([len(intent_relabel)-sequence_length, sequence_length*13])
for i in range(sequence_length):
  input[:,13*i:13*(i+1)] = input_matrix[sequence_length-i-1:len(intent_relabel)-i-1,:]

json_file = open('/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
model.load_weights('/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.h5')


scores = model.evaluate(input, intent_classes[sequence_length:len(intent_relabel)], batch_size = 10, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
