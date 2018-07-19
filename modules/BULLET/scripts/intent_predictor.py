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

files = ['0101', '0102', '0105', '0203', '0204', '0205']
count = 0
for file in files:
    file_str = '../bin/resources/data/'+ file +'.csv'
    data = pd.read_csv(file_str, delim_whitespace = True)

    intent = data.values[0:len(data):100,35]
    gripper_pos = data.values[0:len(data):100,36:39]
    gripper_rot = data.values[0:len(data):100,39:48]
    gripper_angle = data.values[0:len(data):100,48]

    args = np.argwhere(intent != 0)
    intent_relabel = intent[args]
    gripper_pos_relabel = gripper_pos[args[:,0], :]
    gripper_rot_relabel = gripper_rot[args[:,0], :]
    gripper_angle_relabel = gripper_angle[args]

    intent_classes  = np_utils.to_categorical(intent_relabel-1, 4)

    input_matrix = np.hstack([gripper_pos_relabel, gripper_rot_relabel, gripper_angle_relabel])

    sequence_length = 10
    target_classes = intent_classes[sequence_length:len(intent_relabel),:]
    input = np.empty([len(intent_relabel)-sequence_length, sequence_length*13])
    for i in range(sequence_length):
        input[:,13*i:13*(i+1)] = input_matrix[sequence_length-i-1:len(intent_relabel)-i-1,:]

    if count ==0:
        input_concatenate = input
        target_classes_concatenate = target_classes
    else:
        input_concatenate =  np.vstack([input_concatenate, input])
        target_classes_concatenate = np.vstack([target_classes_concatenate, target_classes])

    print input_concatenate.shape, target_classes_concatenate.shape
    count = count+1

model = Sequential()
model.add(Dense(13*sequence_length,input_dim=13*sequence_length, activation='linear'))
model.add(Dense(200, activation = 'tanh'))
model.add(Dense(100, activation = 'tanh'))
# model.add(Dense(50, activation = 'tanh'))
# model.add(Dense(10, activation = 'tanh'))
model.add(Dense(4, kernel_initializer='zeros', bias_initializer='zeros', activation='softmax'))
# model.compile(loss='mse',
#           optimizer='sgd',
#           metrics=['mse'])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

Input,Target = shuffle(input_concatenate,target_classes_concatenate, random_state=2) # shuffling data for randomness
X_train, X_test, Y_train, Y_test = train_test_split(Input, Target, test_size=0.2, random_state=4)

# print input[:,0], input[:,13], input[:,26]
model.fit(X_train, Y_train , epochs = 30, batch_size = 10)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
with open("/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.h5")
print("Model Saved to Disk Sucessfully!")

    # print len(intent_relabel), input
