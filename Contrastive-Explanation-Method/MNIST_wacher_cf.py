import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from time import time
from alibi.explainers import CounterFactual
import cv2
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
plt.gray()
plt.imshow(x_test[1])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

xmin, xmax = -.5, .5
x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin


def cnn_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
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
# cnn.fit(x_train, y_train, batch_size=64, epochs=3, verbose=0)
# cnn.save('./models/Wacher_mnist_cnn.h5')

cnn = load_model('./models/Wacher_mnist_cnn.h5')
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])


nbr_image = 600

path = './datasets/MNIST/'
test_image = np.zeros((nbr_image,28,28))

for i in range(nbr_image):
    test_image[i] = cv2.imread(path+str(i)+'.jpeg', cv2.IMREAD_GRAYSCALE).reshape(1,28,28)/255

test_min = test_image.min()
test_max = test_image.max()
# test_image = ((test_image - test_min) / (test_max - test_min)) * (xmax - xmin) + xmin
test_image = test_image-0.5
test_image = np.reshape(test_image, test_image.shape + (1,))

for nbr_experiments in range(nbr_image):
    

    X = test_image[nbr_experiments].reshape((1,) + test_image[nbr_experiments].shape)
    plt.imshow(X.reshape(28, 28))


    shape = (1,) + test_image.shape[1:]
    target_proba = 0.9
    tol = 0.01 # want counterfactuals with p(class)>0.99
    target_class = np.argsort(-cnn.predict(X)[0])[1] # any class other than 7 will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (test_image.min(),test_image.max())

    cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                        target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                        max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                        feature_range=feature_range)

    start_time = time()
    explanation = cf.explain(X)
    print('Explanation took {:.3f} sec'.format(time() - start_time))

    pred_class = explanation.cf['class']
    proba = explanation.cf['proba'][0][pred_class]

    print(f'Counterfactual prediction: {pred_class} with probability {proba}')
    plt.imshow(explanation.cf['X'].reshape(28, 28))
    
    cf_tmp = explanation.cf['X'].reshape(28, 28)
    # cf_tmp = (cf_tmp+xmin) * (xmax-xmin) / (test_max-test_min) + test_min
    cf_tmp = cf_tmp+0.5
    save_to_image = np.array(cf_tmp*255, dtype='uint8')
    saving_path = './Results/MNIST_result_wacher/'+str(nbr_experiments)+'_'+str(pred_class)+'.jpeg'
    cv2.imwrite(saving_path,save_to_image)
    print(nbr_experiments, "saved to", saving_path)





