import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from alibi.explainers import CEM

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
plt.gray()
plt.imshow(x_test[4])

# x_full = np.append(x_train,x_test, axis=0)

# x_train_9 = x_full[np.where(y_train==9)]
# for i in range(10000):
#     saving_path = './datasets/MNIST_9_full/'+str(i)+'.jpeg'
#     cv2.imwrite(saving_path,x_train_9[i])




x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)


def cnn_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn

# cnn = cnn_model()
# cnn.summary()
# cnn.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)
# cnn.save('./models/CEM_mnist_cnn.h5', save_format='h5')

cnn = load_model('./models/CEM_mnist_cnn.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])

def ae_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

ae = ae_model()
ae.summary()
ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
ae.save('./models/CEM_mnist_ae.h5', save_format='h5')

ae = load_model('./models/CEM_mnist_ae.h5')

decoded_imgs = ae.predict(x_test)
n = 5
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


nbr_image = 600

path = './datasets/MNIST/'
test_image = np.zeros((nbr_image,28,28))

for i in range(nbr_image):
    test_image[i] = cv2.imread(path+str(i)+'.jpeg', cv2.IMREAD_GRAYSCALE).reshape(1,28,28)
test_image = ((test_image - test_image.min()) / (test_image.max() - test_image.min())) * (xmax - xmin) + xmin
test_image = np.reshape(test_image, test_image.shape + (1,))

test_image = test_image.astype('float32') / 255

# idx = 15
# X = x_test[idx].reshape((1,) + x_test[idx].shape)
# plt.imshow(X.reshape(28, 28))
for nbr_experiments in range(nbr_image) :
    X = test_image[nbr_experiments].reshape(1,28,28,1)

    cnn.predict(X).argmax(), cnn.predict(X).max()

    mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + x_train.shape[1:]  # instance shape
    kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
                # class predicted by the original instance and the max probability on the other classes
                # in order for the first loss term to be minimized
    beta = .1  # weight of the L1 loss term
    gamma = 100  # weight of the optional auto-encoder loss term
    c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or
                # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = 10  # nb of updates for c
    max_iterations = 1000  # nb of iterations per value of c
    feature_range = (x_train.min(),x_train.max())  # feature range for the perturbed instance
    clip = (-1000.,1000.)  # gradient clipping
    lr = 1e-2  # initial learning rate
    no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                    # perturbations towards this value means removing features, and away means adding features
                    # for our MNIST images, the background (-0.5) is the least informative,
                    # so positive/negative perturbations imply adding/removing features

    # initialize CEM explainer and explain instance
    cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
            gamma=gamma, ae_model=ae, max_iterations=max_iterations,
            c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

    explanation = cem.explain(X)

    print(f'Pertinent negative prediction: {explanation.PN_pred}')
    plt.imshow(explanation.PN.reshape(28, 28))
    
    save_to_image = np.array(explanation.PN.reshape(28, 28)*255, dtype='uint8')
    saving_path = './Results/MNIST_result_CEM/'+str(nbr_experiments)+'_'+str(explanation.PN_pred)+'.jpeg'
    cv2.imwrite(saving_path,save_to_image)
    print(nbr_experiments, "saved to", saving_path)




