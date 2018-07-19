import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.utils import shuffle
import numpy as np
import pickle
import pandas as pd

data = pd.read_csv('../bin/resources/data/0103.csv', delim_whitespace = True)

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
input = np.empty([len(intent_relabel)-sequence_length, sequence_length*13])
for i in range(sequence_length):
  input[:,13*i:13*(i+1)] = input_matrix[sequence_length-i-1:len(intent_relabel)-i-1,:]

json_file = open('/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
model.load_weights('/home/ankur/chai3d/modules/BULLET/bin/resources/nn_models/model01.h5')

X = input
Y = intent_classes[sequence_length:len(intent_relabel)]

# # Create the test and training data by splitting them
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
# #
# # # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(Y_train, number_of_labels)
# Y_test = np_utils.to_categorical(Y_test, number_of_labels)

# prediction and true labels
y_prob = model.predict(X, batch_size=1000, verbose=0)
y_pred = [np.argmax(prob) for prob in y_prob]
y_true = [np.argmax(prob) for prob in Y]
print "Prediction and True Labels Created"
print y_true[100]

emotion_labels = ['before_pick', 'pick', 'before_place', 'place']

# Create the confusion matrix
print "Creating Confusion"
cm = confusion_matrix(y_true, y_pred)
print cm.shape
print cm
cmap=plt.cm.YlGnBu
fig = plt.figure(figsize=(4,4))
matplotlib.rcParams.update({'font.size': 16})
ax  = fig.add_subplot(111)
matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
fig.colorbar(matrix)
print "Prediction results"
for i in range(0,4):
    for j in range(0,4):
        ax.text(j,i,cm[i,j],va='center', ha='center')
        print cm[i,j]
# ax.set_title('Confusion Matrix')
ticks = np.arange(len(emotion_labels))
ax.set_xticks(ticks)
ax.set_xticklabels(emotion_labels, rotation=45)
ax.set_yticks(ticks)
ax.set_yticklabels(emotion_labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print "Done!"
