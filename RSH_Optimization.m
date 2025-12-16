# How does random search compare against using no hyperparameter tuning when training a CNN on the CIFAR-10 dataset, both in terms of validation accuracy and validation loss?

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import itertools

"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Define the model architecture
def build_model(params):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(params['filters'], (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(params['filters']*2, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params['dense_units'], activation='relu'),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
"""

# Load the dataset (e.g. MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(x_train.shape)
print(x_test.shape)
#x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
#x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Define the model architecture
def build_model(params):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(params['filters'], (3,3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(params['filters']*2, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params['dense_units'], activation='relu'),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

hyperparameter_space_random = {
    'filters': [16,32],
    'dense_units': [64,128],
    'dropout_rate': [ 0.4, 0.2, 0.3],
    'learning_rate': [0.001, 0.01, 0.1]
}

def randomsearch(hyperparameter_space, X_train, y_train, X_val, y_val, num_evaluations=1):
    best_val_acc = 0
    for i in range(num_evaluations):
        hyperparameters = {k: np.random.choice(v) for k, v in dict(hyperparameter_space).items()}
        model = build_model(hyperparameters)
        history= model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=0)
        accuracy = history.history['val_accuracy'][-1] 
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_hyperparameters = hyperparameters
    return best_hyperparameters,best_val_acc,history

best_hyperparameters_random, val_acc_random, random_history = randomsearch(hyperparameter_space_random, x_train, y_train, x_test, y_test)

#Printing the best hyperparameters found by Random Search
print("Best hyperparameters found by Random Search: ",val_acc_random)
print(best_hyperparameters_random)


# Train and evaluate the model with the best hyperparameters with random search

best_model_random = build_model(best_hyperparameters_random)
best_model_random.fit(x_train, y_train,epochs=5,batch_size=32,validation_data=(x_test, y_test))
test_loss_random, test_acc_random = best_model_random.evaluate(x_test, y_test)
print('Test accuracy with random search:', test_acc_random)


params = {'filters': 16, 'dense_units': 64, 'dropout_rate': 0.5}

model_no_hp_opt = build_model(params)
no_hp_history=model_no_hp_opt.fit(x_train, y_train,epochs=5,batch_size=32,validation_data=(x_test, y_test))
test_loss_no_hp_opt, test_acc_no_hp_opt = model_no_hp_opt.evaluate(x_test, y_test)
print('Test accuracy with default hyperparameters:', test_acc_no_hp_opt)

print("random accuracy",random_history.history['val_accuracy'])
print("grid accuracy",no_hp_history.history['val_accuracy'])


plt.figure(1)
plt.plot(random_history.history['val_accuracy'], label='Random Accuracy')
plt.plot(no_hp_history.history['val_accuracy'], label='No Hyperparameter Optimization Accuracy')

plt.title('Validation Accuracy Comparison')
plt.xlabel('Evaluation')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

print("random accuracy",random_history.history['val_loss'])
print("grid accuracy",no_hp_history.history['val_loss'])

plt.figure(2)
plt.plot(random_history.history['val_loss'], label='Random loss')
plt.plot(no_hp_history.history['val_loss'], label='No Hyperparameter Optimization loss')

plt.title('Validation loss Comparison')
plt.xlabel('Evaluation')
plt.ylabel('Validation loss')
plt.legend()
plt.show()