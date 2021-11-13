---
layout: post
title: "Blog Post 5 about Model Training"
date: 2021-11-11
---
# Blog Post 5 - Model to Distinguish Image Objects
In this blog post, we will learn several new skills and concepts related to image classification in Tensorflow. 

We want to build a machine learning algorithm to distinguish between pictures of dogs and pictures of cats.



First, we want to load packages and obtain data.

## 1. Load Packages and Obtain Data


```python
import matplotlib.pyplot as plt
import numpy as np

import os
from tensorflow.keras import utils 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import collections
```

After loading packages, we need to access the data. We will use a sample data set provided by the TensorFlow team that contains labeled images of cats and dogs.


```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.


Now, we have created TensorFlow Datasets for training, validation, and testing.

## Working with Datasets

We define a function `two_row_visualization` to show three random pictures of cates in the first row and three random pictures of dogs in the second row.

Hint: `take` method is useful to get a piece of the dataset.


```python
def two_row_visualization(train_dataset):
  class_names = train_dataset.class_names

  plt.figure(figsize=(10, 10))
  dataset = train_dataset.take(1)
  
  for images, labels in dataset:
    cat = []
    dog = []
    for i in range(len(labels)):
      # determine the index of cat images and dog images
      # add the index to the corresponding list
      if (len(cat)<3):
        if (class_names[labels[i]]=="cats"):
          cat.append(i)
      if (len(dog)<3):
        if (class_names[labels[i]]=="dogs"):
          dog.append(i)
      if (len(cat) == 3) and (len(dog) == 3):
        break
    index = cat + dog

    # plot the images
    for i in range(6):
      ax = plt.subplot(2, 3, i + 1)
      plt.imshow(images[index[i]].numpy().astype("uint8"))
      plt.title(class_names[labels[index[i]]])
      plt.axis("off")
```

Run the function below.


```python
two_row_visualization(train_dataset)
```


    
![output_9_0.png](/images/output_9_0.png)
    


To rapidly read data, we need the code chunk below.


```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

## Check Label Frequencies

First, we create an iterator called `labels`.



```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()

```

We now want to compute the number of dog and cat images respectively in the training data. Label 0 corresponds to "cats", and label 1 corresponds to "dogs".

Hint: We use `collections.Counter()` to compute the number.


```python
# Put the label into the list in order to apply the Counter method
counts=collections.Counter(list(labels_iterator))
counts
```




    Counter({0: 1000, 1: 1000})



Since the number of cat images and dog images are the same, the baseline model has the accuracy of 50% in our case.

## First Model

Create a tf.keras.Sequential model using some of the layers. We will include at least two `Conv2D` layers, at least two `MaxPooling2D` layers, at least one `Flatten` layer, at least one `Dense` layer, and at least one `Dropout` layer. 

We will train the model and plot the history of accuracy on both training and validation dataset.

We will build the model first. Call it `model1`.


```python
model1 = models.Sequential([
    # beginning (input)
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(160, 160, 3)),
    # setting activation as "relu" to apply a nonlinear transformation
    
    layers.MaxPooling2D((3, 3)),
    # MaxPooling takes the maximum value over an input window 
    # We set the window shape as 2 by 2
    
    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((3, 3)),

    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),  

    layers.Dropout(0.2),
    # Dropout helps to prevent the overfitting

    # end (output)
    layers.Flatten(), # reshape from 2d to 1d
    layers.Dense(64, activation='relu'),
    layers.Dense(2), # number of classes
])
```

Train the model.


When training the model, we set `from_logits` as `True`, which says compute the softmax operation when evaluating the loss function.

Once we have compiled the model, we prepare for the plotting of accuracy history.


```python
model1.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data= validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 7s 79ms/step - loss: 6.1894 - accuracy: 0.5230 - val_loss: 0.7189 - val_accuracy: 0.5792
    Epoch 2/20
    63/63 [==============================] - 5s 72ms/step - loss: 0.6756 - accuracy: 0.6060 - val_loss: 0.6565 - val_accuracy: 0.6163
    Epoch 3/20
    63/63 [==============================] - 5s 73ms/step - loss: 0.6000 - accuracy: 0.6785 - val_loss: 0.6355 - val_accuracy: 0.6757
    Epoch 4/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5393 - accuracy: 0.7310 - val_loss: 0.6730 - val_accuracy: 0.6460
    Epoch 5/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.4821 - accuracy: 0.7725 - val_loss: 0.6666 - val_accuracy: 0.6757
    Epoch 6/20
    63/63 [==============================] - 5s 73ms/step - loss: 0.4315 - accuracy: 0.8005 - val_loss: 0.7127 - val_accuracy: 0.6522
    Epoch 7/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.4157 - accuracy: 0.8040 - val_loss: 0.6952 - val_accuracy: 0.6634
    Epoch 8/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.3641 - accuracy: 0.8320 - val_loss: 0.6739 - val_accuracy: 0.6844
    Epoch 9/20
    63/63 [==============================] - 5s 79ms/step - loss: 0.3337 - accuracy: 0.8585 - val_loss: 0.7342 - val_accuracy: 0.7042
    Epoch 10/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.3077 - accuracy: 0.8635 - val_loss: 0.7628 - val_accuracy: 0.6906
    Epoch 11/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.2865 - accuracy: 0.8770 - val_loss: 0.7991 - val_accuracy: 0.6894
    Epoch 12/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.2361 - accuracy: 0.9025 - val_loss: 0.8190 - val_accuracy: 0.6894
    Epoch 13/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.2033 - accuracy: 0.9135 - val_loss: 0.8781 - val_accuracy: 0.6869
    Epoch 14/20
    63/63 [==============================] - 5s 72ms/step - loss: 0.2290 - accuracy: 0.9115 - val_loss: 0.9032 - val_accuracy: 0.6869
    Epoch 15/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.2228 - accuracy: 0.9115 - val_loss: 0.8553 - val_accuracy: 0.6881
    Epoch 16/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.1736 - accuracy: 0.9220 - val_loss: 0.9175 - val_accuracy: 0.7079
    Epoch 17/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.1509 - accuracy: 0.9415 - val_loss: 1.0585 - val_accuracy: 0.6782
    Epoch 18/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.1453 - accuracy: 0.9425 - val_loss: 0.9598 - val_accuracy: 0.7215
    Epoch 19/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.1042 - accuracy: 0.9590 - val_loss: 1.0405 - val_accuracy: 0.6993
    Epoch 20/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.0940 - accuracy: 0.9610 - val_loss: 1.1012 - val_accuracy: 0.7017


Now, we want to plot the accuracy history.


```python
# plot the traning data accuracy
plt.plot(history.history["accuracy"], label = "training")
# plot the validation data accuracy
plt.plot(history.history["val_accuracy"], label = "validation")
# set axis
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fd915762ed0>




    
![output_23_1.png](/images/output_23_1.png)


**The accuracy of my model1 stabilized between 57% and 70% during training.**

* What I did:

  I change the number of `Conv2D` layers and three `MaxPooling2D` layers, the kernel size of `Conv2D`, and the window shape size of `MaxPooling2D` achieve a stable accuracy of more than 52%. The model performed better than the baseline model by 7% to 20%.

* Overfitting:

  By looking at the plot, we could see the training accuracy is much higher than the validation accuracy. This indicates the overfitting of model 1.

## Model with Data Augmentation

Data augmentation refers to the practice of including modified copies of the same image in the training set. By implementing data augmentation, we could help our model learn better about invariant features of the input images.

We will create a `tf.keras.layers.RandomFlip()` layer to flip our image.




```python
flip = tf.keras.layers.RandomFlip()

for image, _ in train_dataset.take(1):
  plt.figure(figsize = (10,10))
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    flipped = flip(tf.expand_dims(image[0],axis = 0)) 
    plt.imshow(flipped[0]/255)
    plt.axis("off")
```


    
![output_26_0.png](/images/output_26_0.png)
    


Next, we create a `tf.keras.layers.RandomRotation()` layer to rotate our image.


```python
rotate = tf.keras.layers.RandomRotation(factor = 0.2)

for image, _ in train_dataset.take(1):
  plt.figure(figsize = (10,10))
  
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    rotated = rotate(tf.expand_dims(image[0],axis = 0)) 
    plt.imshow(rotated[0]/255)
    plt.axis("off")
  
```


    
![output_28_1.png](/images/output_28_0.png)
    


Now, we are gonna to create a `model2`, in which the first two layers are augmentation layers.


```python
model2 = models.Sequential([
    # first add the augmentation layers
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomRotation(factor = 0.2),

    # The following layers are the same as Model1 layers
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(160, 160, 3)),
    
    layers.MaxPooling2D((3, 3)),
    
    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((3, 3)),  
  
    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),  

    layers.Dropout(0.2),
   
    layers.Flatten(), # reshape from 2d to 1d
    layers.Dense(64, activation='relu'),
    layers.Dense(2), # number of classes
])
```

Train model 2.


```python
model2.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data= validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 7s 82ms/step - loss: 16.4690 - accuracy: 0.4960 - val_loss: 0.7432 - val_accuracy: 0.5347
    Epoch 2/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.7195 - accuracy: 0.5440 - val_loss: 0.7168 - val_accuracy: 0.5594
    Epoch 3/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6966 - accuracy: 0.5620 - val_loss: 0.7087 - val_accuracy: 0.5854
    Epoch 4/20
    63/63 [==============================] - 5s 73ms/step - loss: 0.6909 - accuracy: 0.5735 - val_loss: 0.6976 - val_accuracy: 0.5804
    Epoch 5/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6861 - accuracy: 0.5855 - val_loss: 0.6730 - val_accuracy: 0.6064
    Epoch 6/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6726 - accuracy: 0.5880 - val_loss: 0.6608 - val_accuracy: 0.6101
    Epoch 7/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6743 - accuracy: 0.5910 - val_loss: 0.6602 - val_accuracy: 0.5866
    Epoch 8/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6487 - accuracy: 0.6195 - val_loss: 0.6573 - val_accuracy: 0.6101
    Epoch 9/20
    63/63 [==============================] - 5s 73ms/step - loss: 0.6511 - accuracy: 0.6175 - val_loss: 0.6829 - val_accuracy: 0.5879
    Epoch 10/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6478 - accuracy: 0.6215 - val_loss: 0.6711 - val_accuracy: 0.6040
    Epoch 11/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6579 - accuracy: 0.6065 - val_loss: 0.6611 - val_accuracy: 0.6114
    Epoch 12/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6502 - accuracy: 0.6135 - val_loss: 0.6633 - val_accuracy: 0.6052
    Epoch 13/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6432 - accuracy: 0.6245 - val_loss: 0.6664 - val_accuracy: 0.5891
    Epoch 14/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6387 - accuracy: 0.6160 - val_loss: 0.6669 - val_accuracy: 0.5953
    Epoch 15/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6326 - accuracy: 0.6345 - val_loss: 0.6638 - val_accuracy: 0.6151
    Epoch 16/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6351 - accuracy: 0.6300 - val_loss: 0.6727 - val_accuracy: 0.6176
    Epoch 17/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.6292 - accuracy: 0.6385 - val_loss: 0.6696 - val_accuracy: 0.5978
    Epoch 18/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.6293 - accuracy: 0.6460 - val_loss: 0.6906 - val_accuracy: 0.5879
    Epoch 19/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.6349 - accuracy: 0.6275 - val_loss: 0.6639 - val_accuracy: 0.6089
    Epoch 20/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.6295 - accuracy: 0.6325 - val_loss: 0.6574 - val_accuracy: 0.6349


Now, we want to plot the accuracy history of model2.


```python
# plot the traning data accuracy
plt.plot(history.history["accuracy"], label = "training")
# plot the validation data accuracy
plt.plot(history.history["val_accuracy"], label = "validation")
# set axis
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fd898793050>




    
![output_34_1.png](/images/output_34_1.png)
    


**The accuracy of my model2 stabilized between 55% and 63% during training.**

* What I changed:

  I added two augmentation layers to model1.  The rest of the layers are the same. The model performed a bit worse than model1 by 2% to 7%.

* Overfitting:

  By looking at the plot, we could see the training accuracy and the validation accuracy become really close at the end. This indicates the overfitting problem of model 1 has been solved by adding augmentation layers in model 2.

## Data Preprocessing

In this part, we are going to create a preprocessor layer to increase the model performance score.


```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```

Then, we incorporate the preprocessor layer into our `model2` as the very first layer, and we call the new model `model3`.


```python
model3 = models.Sequential([
    # the preprocessor layer
    preprocessor,
    # augmentation layer
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomRotation(factor = 0.2),

    # The following layers are the same as Model1 layers
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(160, 160, 3)),
    
    layers.MaxPooling2D((3, 3)),
    
    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((3, 3)),  

    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),  

    layers.Dropout(0.2),
   
    layers.Flatten(), # reshape from 2d to 1d
    layers.Dense(64, activation='relu'),
    layers.Dense(2), # number of classes
])
```

Train model 3


```python
model3.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data= validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 6s 80ms/step - loss: 0.7016 - accuracy: 0.5175 - val_loss: 0.6510 - val_accuracy: 0.5965
    Epoch 2/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.6591 - accuracy: 0.5940 - val_loss: 0.6309 - val_accuracy: 0.6077
    Epoch 3/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.6298 - accuracy: 0.6345 - val_loss: 0.5971 - val_accuracy: 0.6646
    Epoch 4/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.6136 - accuracy: 0.6525 - val_loss: 0.5882 - val_accuracy: 0.6807
    Epoch 5/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5874 - accuracy: 0.6840 - val_loss: 0.5688 - val_accuracy: 0.6782
    Epoch 6/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5784 - accuracy: 0.6910 - val_loss: 0.5615 - val_accuracy: 0.6968
    Epoch 7/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5880 - accuracy: 0.6830 - val_loss: 0.5563 - val_accuracy: 0.7092
    Epoch 8/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5668 - accuracy: 0.7000 - val_loss: 0.5420 - val_accuracy: 0.7079
    Epoch 9/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.5602 - accuracy: 0.7060 - val_loss: 0.5603 - val_accuracy: 0.7005
    Epoch 10/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5555 - accuracy: 0.7015 - val_loss: 0.5859 - val_accuracy: 0.6832
    Epoch 11/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5414 - accuracy: 0.7225 - val_loss: 0.5244 - val_accuracy: 0.7277
    Epoch 12/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5416 - accuracy: 0.7240 - val_loss: 0.5558 - val_accuracy: 0.7191
    Epoch 13/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5262 - accuracy: 0.7225 - val_loss: 0.5190 - val_accuracy: 0.7438
    Epoch 14/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.5293 - accuracy: 0.7335 - val_loss: 0.5016 - val_accuracy: 0.7463
    Epoch 15/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5242 - accuracy: 0.7305 - val_loss: 0.4984 - val_accuracy: 0.7537
    Epoch 16/20
    63/63 [==============================] - 5s 74ms/step - loss: 0.5110 - accuracy: 0.7410 - val_loss: 0.5223 - val_accuracy: 0.7376
    Epoch 17/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5077 - accuracy: 0.7455 - val_loss: 0.5193 - val_accuracy: 0.7512
    Epoch 18/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.5066 - accuracy: 0.7435 - val_loss: 0.4975 - val_accuracy: 0.7574
    Epoch 19/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.4978 - accuracy: 0.7580 - val_loss: 0.5041 - val_accuracy: 0.7537
    Epoch 20/20
    63/63 [==============================] - 5s 75ms/step - loss: 0.4974 - accuracy: 0.7455 - val_loss: 0.5042 - val_accuracy: 0.7587


Then, we plot the accuracy history of model3.


```python
# plot the traning data accuracy
plt.plot(history.history["accuracy"], label = "training")
# plot the validation data accuracy
plt.plot(history.history["val_accuracy"], label = "validation")
# set axis
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fd822e6cd50>




    
![output_43_1.png](/images/output_43_1.png)
    


**The accuracy of my model3 stabilized between 66% and 75% during training.**

* What I changed:

  I added the prepossessor layer to model2.  The rest of the layers are the same as model2. The model performed better than model1 by around 5%.

* Overfitting:

  By looking at the plot, we could see the training accuracy and the validation accuracy are close. This indicates there is no overfitting in model 3.

## Transfer Learning

In this section, we will incorporate an exisiting model into a full model for our current task.

We will use the base model `MobileNetV2` and configue it as a layer to be incorporated into our model.

The code chunk below first downloads `MobileNetV2`.


```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
    9412608/9406464 [==============================] - 0s 0us/step
    9420800/9406464 [==============================] - 0s 0us/step


Then, we start to create a `model4`. This model uses the following layers:

* The preprocessor layer from Part §4.
* The data augmentation layers from Part §3.
* The base_model_layer constructed above.
* The layers constructed in our first model.


```python
model4 = models.Sequential([
    # preprocessor layer
    preprocessor,
    # augmentation layers
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomRotation(factor = 0.2),

    # incorporate the base model layer
    base_model_layer,

    layers.Dropout(0.2),
    layers.GlobalMaxPool2D(),
    
    layers.Dense(2), # number of classes
])
```


```python
model4.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model (Functional)          (None, 160, 160, 3)       0         
                                                                     
     random_flip_7 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     random_rotation_7 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     model_1 (Functional)        (None, 5, 5, 1280)        2257984   
                                                                     
     dropout_9 (Dropout)         (None, 5, 5, 1280)        0         
                                                                     
     global_max_pooling2d_2 (Glo  (None, 1280)             0         
     balMaxPooling2D)                                                
                                                                     
     dense_16 (Dense)            (None, 2)                 2562      
                                                                     
    =================================================================
    Total params: 2,260,546
    Trainable params: 2,562
    Non-trainable params: 2,257,984
    _________________________________________________________________


By looking at the summary, we could see that model 4 train 2,260,546 parameters. We can see that there is a lot of complexity hidden in the `base_model_layer`.

Train the model 4.


```python
model4.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data= validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 11s 115ms/step - loss: 0.8664 - accuracy: 0.7945 - val_loss: 0.1334 - val_accuracy: 0.9554
    Epoch 2/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.5008 - accuracy: 0.8650 - val_loss: 0.2369 - val_accuracy: 0.9245
    Epoch 3/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.4151 - accuracy: 0.8920 - val_loss: 0.1299 - val_accuracy: 0.9604
    Epoch 4/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.3866 - accuracy: 0.8985 - val_loss: 0.1218 - val_accuracy: 0.9604
    Epoch 5/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2851 - accuracy: 0.9170 - val_loss: 0.1107 - val_accuracy: 0.9604
    Epoch 6/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3405 - accuracy: 0.9080 - val_loss: 0.1118 - val_accuracy: 0.9666
    Epoch 7/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3106 - accuracy: 0.9160 - val_loss: 0.0981 - val_accuracy: 0.9666
    Epoch 8/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2328 - accuracy: 0.9210 - val_loss: 0.1233 - val_accuracy: 0.9567
    Epoch 9/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2921 - accuracy: 0.9185 - val_loss: 0.0991 - val_accuracy: 0.9691
    Epoch 10/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2726 - accuracy: 0.9140 - val_loss: 0.1096 - val_accuracy: 0.9653
    Epoch 11/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.2811 - accuracy: 0.9140 - val_loss: 0.1002 - val_accuracy: 0.9629
    Epoch 12/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3126 - accuracy: 0.9080 - val_loss: 0.1299 - val_accuracy: 0.9542
    Epoch 13/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.2531 - accuracy: 0.9285 - val_loss: 0.1347 - val_accuracy: 0.9505
    Epoch 14/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3014 - accuracy: 0.9110 - val_loss: 0.1198 - val_accuracy: 0.9554
    Epoch 15/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2271 - accuracy: 0.9275 - val_loss: 0.1351 - val_accuracy: 0.9517
    Epoch 16/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2670 - accuracy: 0.9180 - val_loss: 0.1149 - val_accuracy: 0.9542
    Epoch 17/20
    63/63 [==============================] - 6s 95ms/step - loss: 0.2336 - accuracy: 0.9280 - val_loss: 0.1027 - val_accuracy: 0.9641
    Epoch 18/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2161 - accuracy: 0.9280 - val_loss: 0.1124 - val_accuracy: 0.9592
    Epoch 19/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2131 - accuracy: 0.9345 - val_loss: 0.0930 - val_accuracy: 0.9678
    Epoch 20/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2253 - accuracy: 0.9295 - val_loss: 0.0954 - val_accuracy: 0.9653


Then, we plot the accuracy history of model 4.


```python
# plot the traning data accuracy
plt.plot(history.history["accuracy"], label = "training")
# plot the validation data accuracy
plt.plot(history.history["val_accuracy"], label = "validation")
# set axis
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fd820e8af90>




    
![output_54_1.png](/images/output_54_1.png)
    


**The accuracy of my model4 stabilized between 95% and 96% during training.**

* Comparison:

  The model performed much better than all models before by stabalizing an accuracy at around 95%.

* Overfitting:

  By looking at the plot, we could see the validation accuracy is consistently higher than the training accuracy. This indicates there is no overfitting in model 4.

## 6. Score on Test Data
In this section, we are going to evaluate the accuracy of the most performant model, `model4`, on the unseen `test_dataset`.



```python
score = model4.evaluate(test_dataset)

# the second element of the score demonstrates the accuracy of model 4.
print(f"accuracy:", score[1])
```

    6/6 [==============================] - 1s 68ms/step - loss: 0.0859 - accuracy: 0.9740
    accuracy: 0.9739583134651184


We could see that model 4 acquires an accuracy of 97% on the test dataset! Impressive performance!
