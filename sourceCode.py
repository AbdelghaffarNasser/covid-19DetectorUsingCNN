import pandas as pd
import shutil
import os


# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

#TRAIN_PATH = "CovidDataset/Train"
#VAL_PATH = "CodidDataset/Test"

#CNN model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam',metrics=['accuracy'])

model.summary()
    
# treain the machen
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range= 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
    
    )
test_dataset = image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')


train_generator.class_indices

validation_generator = test_dataset.flow_from_directory(
    'CovidDataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2)

model.save("model_adv.h5")
model.evaluate_generator(train_generator)



model = load_model('model_adv.h5')

train_generator.class_indices

y_actual = []
y_test = []

for i in os.listdir("./CovidDataset/Val/Normal/"):
    img = image.load_img("./CovidDataset/Val/Normal/"+i,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
   # plt.imshow(img.reshape(3,224,224))
    p = model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(1)
   
for i in os.listdir("./CovidDataset/Val/Covid/"):
    img = image.load_img("./CovidDataset/Val/Covid/"+i,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    p = model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(0)
    
y_actual = np.array(y_actual)
y_test = np.array(y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_actual, y_test)

import seaborn as sn
import pandas as pd

df_cm = pd.DataFrame(cm, range(2), range(2))
#plt.figure(figsize = (5,3))
sn.heatmap(df_cm, annot=True)


for i in os.listdir("./CovidDataset/Detect/"):
    img = image.load_img("./CovidDataset/Detect/"+i,target_size=(224,224))
    img1 = image.img_to_array(img)
    img2 = np.expand_dims(img1,axis=0)
    p = model.predict_classes(img2)
    if p==0:
        imgplot = plt.imshow(img1[:,:,0],cmap="hot")
        plt.title('Covid')
        
    else:
         imgplot = plt.imshow(img1[:,:,0])
         plt.title('Normal')