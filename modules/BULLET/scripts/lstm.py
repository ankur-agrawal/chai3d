import pandas as pd
import numpy as np
import keras
import tensorflow
import h5py
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import sgd
from keras.optimizers import Adam
from keras.utils import np_utils
# from keras.preprocessing.sequence import TimeSeriesGenerator
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler

class Intent_Predictor(object):
    def __init__(self):
        self.model = Sequential()
    def load_model(self, sequence_length):
        self.model.add(Dense(13,input_shape=(10,13), activation='linear'))
        self.model.add(LSTM(5, return_sequences = True, dropout = 0.1))
        self.model.add(LSTM(4, return_sequences = True, activation = 'softmax'))
        # self.model.add(TimeDistributed(Dense(4)))
        # self.model.add(Activation('softmax'))
        #self.model.add(Dense(5, activation = 'relu'))
        # self.model.add(Dense(50, activation = 'tanh'))
        # self.model.add(Dense(10, activation = 'tanh'))
        # self.model.add(Dense(4, output_shape = (10,4), kernel_initializer='zeros', bias_initializer='zeros', activation='softmax'))
    def step_decay(epochs):
        initial_lrate = 0.000001
    	drop = 0.6
    	epochs_drop = 100
    	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    	return lrate
    def compile_model(self):
        adam = Adam(lr=0.0001, beta_1=0.999, beta_2=0.999, epsilon=1e-07, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # self.model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
        self.model.summary()
        lrate= LearningRateScheduler(self.step_decay)
        # callbacks_list=[lrate]

def load_data(Train_Files, Test_Files, time_steps, sequence_length):
    count_train = 0
    for file in Train_Files:
        file_str = '../bin/resources/data/'+ file +'.csv'
        data_train = pd.read_csv(file_str, delim_whitespace = True)

        intent_train = data_train.values[0:len(data_train):time_steps,35]
        gripper_pos_train = data_train.values[0:len(data_train):time_steps,36:39]
        gripper_rot_train = data_train.values[0:len(data_train):time_steps,39:48]
        gripper_angle_train = data_train.values[0:len(data_train):time_steps,48]

        args_train = np.argwhere(intent_train != 0)
        intent_relabel_train = intent_train[args_train] - 1
        gripper_pos_relabel_train = gripper_pos_train[args_train[:,0], :]
        gripper_rot_relabel_train = gripper_rot_train[args_train[:,0], :]
        gripper_angle_relabel_train = gripper_angle_train[args_train]

        intent_classes_train  = np_utils.to_categorical(intent_relabel_train, 4)
        input_matrix_train = np.hstack([gripper_pos_relabel_train, gripper_rot_relabel_train, gripper_angle_relabel_train])

        # sequence = TimeseriesGenerator(input_matrix_train, intent_classes_train, length, sampling_rate=1, stride=2, start_index=0, end_index=None, shuffle=False, reverse=False)

        # print sequence
        # target_classes_train = intent_classes_train[sequence_length:len(intent_relabel_train),:]

        input_train = np.empty([len(intent_relabel_train)-sequence_length, sequence_length,13])
        target_classes_train = np.empty([len(intent_relabel_train)-sequence_length, sequence_length,4])
        for i in range(sequence_length):
            input_train[:,i,:] = input_matrix_train[sequence_length-i-1:len(intent_relabel_train)-i-1,:]
            target_classes_train[:,i,:] = intent_classes_train[sequence_length-i-1:len(intent_relabel_train)-i-1,:]
        if count_train ==0:
            input_concatenate_train = input_train
            target_classes_concatenate_train = target_classes_train
        else:
            input_concatenate_train =  np.vstack([input_concatenate_train, input_train])
            target_classes_concatenate_train = np.vstack([target_classes_concatenate_train, target_classes_train])

        count_train = count_train+1
    print "Training Data Size"
    print "Inputs:", input_concatenate_train.shape
    print "Targets:", target_classes_concatenate_train.shape
    X_train,Y_train = shuffle(input_concatenate_train,target_classes_concatenate_train, random_state=2)

    count_test = 0
    for file in Test_Files:
        file_str = '../bin/resources/data/'+ file +'.csv'
        data_test = pd.read_csv(file_str, delim_whitespace = True)

        intent_test = data_test.values[0:len(data_test):time_steps,35]
        gripper_pos_test = data_test.values[0:len(data_test):time_steps,36:39]
        gripper_rot_test = data_test.values[0:len(data_test):time_steps,39:48]
        gripper_angle_test = data_test.values[0:len(data_test):time_steps,48]

        args_test = np.argwhere(intent_test != 0)
        intent_relabel_test = intent_test[args_test] - 1
        gripper_pos_relabel_test = gripper_pos_test[args_test[:,0], :]
        gripper_rot_relabel_test = gripper_rot_test[args_test[:,0], :]
        gripper_angle_relabel_test = gripper_angle_test[args_test]

        intent_classes_test  = np_utils.to_categorical(intent_relabel_test, 4)
        input_matrix_test = np.hstack([gripper_pos_relabel_test, gripper_rot_relabel_test, gripper_angle_relabel_test])
        # target_classes_test = intent_classes_test[sequence_length:len(intent_relabel_test),:]

        input_test = np.empty([len(intent_relabel_test)-sequence_length, sequence_length,13])
        target_classes_test = np.empty([len(intent_relabel_test)-sequence_length, sequence_length,4])
        for i in range(sequence_length):
            input_test[:,i,:] = input_matrix_test[sequence_length-i-1:len(intent_relabel_test)-i-1,:]
            target_classes_test[:,i,:] = intent_classes_test[sequence_length-i-1:len(intent_relabel_test)-i-1,:]

        if count_test ==0:
            input_concatenate_test = input_test
            target_classes_concatenate_test = target_classes_test
        else:
            input_concatenate_test =  np.vstack([input_concatenate_test, input_test])
            target_classes_concatenate_test = np.vstack([target_classes_concatenate_test, target_classes_test])

        count_test = count_test+1

    print "Testing Data Size"
    print "Inputs:" , input_concatenate_test.shape
    print "Targets:", target_classes_concatenate_test.shape
    X_test,Y_test = shuffle(input_concatenate_test,target_classes_concatenate_test, random_state=2)

    return X_train, Y_train, X_test, Y_test

train_files = ['0101', '0102','0105', '0202', '0205', '0204']
test_files = ['0104', '0203']

# load_data(train_files, test_files, 100, 10)

x_train, y_train, x_test, y_test = load_data(train_files, test_files, 100, 10)

nn_model = Intent_Predictor()

nn_model.load_model(10)
nn_model.compile_model()

# X_train, X_test, Y_train, Y_test = train_test_split(Input, Target, test_size=0.2, random_state=4)

history = nn_model.model.fit(x_train, y_train , validation_data=(x_test, y_test), epochs = 200, batch_size = 50)
scores = nn_model.model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



print(history.history.keys())
#  accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
