import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import random
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

def model_gen(n_layers):
    in1 = tf.keras.layers.Input(shape=(1,))
    dense_layers = [0 for i in range(n_layers)]
    dot_layers = [in1 for i in range(n_layers)]
    
    for i in range(1,n_layers,1):
        dense_layers[i] = tf.keras.layers.Dense(1, kernel_initializer='random_uniform', 
                    bias_initializer='random_uniform', activation='linear')(dot_layers[i-1])
        dot_layers[i] = tf.keras.layers.multiply([dense_layers[i], in1])
   
    out =  tf.keras.layers.Dense(1, kernel_initializer='random_uniform', 
                             bias_initializer='random_uniform', activation='linear')(dot_layers[-1])

    model = tf.keras.models.Model(inputs=in1, outputs=out)
    return model
  
def data_gen(func,sample_size=100):
    train_x = [[random()*np.pi] for i in range(sample_size)]
    train_y = [[func(x[0])] for x in train_x]
    return train_x, train_y

model = model_gen(3)
train_x, train_y = data_gen(np.cos,sample_size=100)

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=500, verbose=False)


test_x = [[random()*np.pi] for i in range(100)]
plt.plot(train_x,train_y,'o',label='training')
plt.plot(test_x,[model.predict(x)[0] for x in test_x],'o',label='trained')
plt.legend(loc="upper right")
plt.ylim(-2, 2.0)
plt.show()
