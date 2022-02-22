# MNIST - Convolution Neural Network

## MNIST dataset

**IMPORTANT: This repository has been moved to https://gitlab.com/datapythonista/mnist**

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

![](https://github.com/datapythonista/mnist/raw/master/img/samples.png)

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

There are four files available, which contain separately train and test, and images and labels.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.

## Usage

mnist makes it easier to download and parse MNIST files.

To automatically download the train files, and display the first image in the
dataset, you can simply use:

```python
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# TF already did train test for us, we need to do tuple unpacking
(X_train, y_train),(X_test,y_test) = mnist.load_data()

single_image = X_train[0]
plt.imshow(single_image, cmap='binary')
```

![](https://github.com/datapythonista/mnist/raw/master/img/img_5.png)

The dataset is downloaded and cached in your temporary directory, so, calling
the functions again, is much faster and doesn't hit the server.

Images are returned as a 3D numpy array (samples * rows * columns). To train
machine learning models, usually a 2D array is used (samples * features) [WHICH WILL HAPPEN IN FLATTEN LAYER IN MODELLING]. To
get it, simply use:

```python
model = Sequential()


# Choose filters in powers of 2, the more the better for complex images
# Strides are small here because the image size is 28x28 only

model.add(Conv2D(filters=32, kernel_size=(4,4),
                input_shape=(28,28,1), activation='relu'))

#pool_size half of kernel
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten image means 28x28 images becomes 784 array
model.add(Flatten())

model.add(Dense(128, activation='relu'))

#Output layer
# activation is softmax because multi-class classification
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
             metrics = ['accuracy']) # find all metrics at 'keras.io/metrics'
```
It supports Python 2.7 and Python >= 3.5.