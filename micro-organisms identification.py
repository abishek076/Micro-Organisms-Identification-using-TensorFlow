import os,glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import randint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.metrics import classification_report
import numpy as np
from os import path, listdir


#  Specify root path
root_path = 'C:/Users/gabis/Downloads/project/Micro_Organism'
name_class = os.listdir(root_path)

dataset_Path = list(glob.glob(root_path+'/**/*.*'))
labels = list(map(lambda x : os.path.split(os.path.split(x)[0])[1],dataset_Path))

dataset_Path = pd.Series(dataset_Path,name = 'FilePath').astype(str)
labels = pd.Series(labels,name='Label')
data = pd.concat([dataset_Path,labels],axis =1)
data = data.sample(frac=1).reset_index(drop= True)
data.head(5)
# data visualization
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i]-10,y[i], ha = 'center')
counts = data.Label.value_counts()

plt.bar(counts.index, counts)
def add_labels(index, y):
    for i in range(len(index)):
        plt.text(i, y.iloc[i]-10, y.iloc[i], ha='center')
add_labels(counts.index, counts)
plt.xticks(rotation=90)
plt.xlabel('Type')
plt.ylabel('Label')
plt.show()

fig,axes = plt.subplots(nrows=5,ncols=3,figsize=(10,8),subplot_kw={'xticks':[],'yticks':[]})
for i ,ax,in enumerate(axes.flat):
  ax.imshow(plt.imread(data.FilePath[i]))
  ax.set_title(data.Label[i])
plt.tight_layout()
plt.show()

# spliting image
train,rem = train_test_split(data,test_size =0.20,random_state = 42 )
test,valid = train_test_split(data,test_size =0.50,random_state = 42 )

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,)
test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_dataframe(dataframe = train,x_col = 'FilePath',y_col = 'Label',target_size=(224,224),
                                              class_mode ='categorical', color_mode='rgb',batch_size =8,shuffle = True,seed =42)
valid_gen = train_datagen.flow_from_dataframe(dataframe = valid,x_col = 'FilePath',y_col = 'Label',target_size=(224,224),
                                              class_mode ='categorical', color_mode='rgb',batch_size =8,shuffle = False,seed =42)
test_gen =  test_datagen.flow_from_dataframe(dataframe = test,x_col = 'FilePath',y_col = 'Label',target_size=(224,224),
                                             class_mode ='categorical', color_mode='rgb',batch_size =8,shuffle = False,seed =42)

NO_CLASSES = max(train_gen.class_indices.values()) + 1
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x) #used to replace fully connected layers in classical CNNs.
#It will generate one feature map for each corresponding category of the classification task in the last mlpcov layer(1 X 1 convolutions).
x = Dense(1024,activation='relu')(x)   # add dense layers so learn more complex functions and classify for better results.
x = Dense(1024,activation='relu')(x)   # dense layer 2
x = Dense(512,activation='relu')(x)    # dense layer 3

preds = Dense(NO_CLASSES,activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = preds) #create a new model with the base model's original input

#don't train the first 19 layers
for layer in model.layers[:19]:
    layer.trainable=False
#train the rest of the layers
for layer in model.layers[19:]:
    layer.trainable=True

# model Architecture
tf.keras.utils.plot_model(model,'model.png',show_shapes=True)

#Compiling Model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#callbacks
my_callbacks = [EarlyStopping(monitor = 'val_accuracy',min_delta=0,patience=2,mode='auto')]
model.fit(train_gen,validation_data = valid_gen,epochs=25,callbacks=[my_callbacks])
model.evaluate(test_gen)

#predict the label of the test_gen
pred = model.predict(test_gen)
pred = np.argmax(pred,axis=1)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v, in labels.items())
pred = [labels[k]for k in pred]
y_test = list(test.Label)
print(classification_report(y_test,pred))
labels = (train_gen.class_indices)
print(labels)

pred = model.predict(test_gen)
pred = np.argmax(pred,axis=1)
labels = dict((v,k) for k,v, in labels.items())
pred = [labels[k]for k in pred]

fig,axes = plt.subplots(nrows=5,ncols=3,figsize=(20,18),subplot_kw={'xticks':[],'yticks':[]})
for i ,ax,in enumerate(axes.flat):
    ax.imshow(plt.imread(test_gen.filenames[i]))
    title =str( f"Class : {labels[test_gen.classes[i]]}\nPred : {pred[i]}\n")
    ax.set_title(title,fontsize=15)
plt.tight_layout()
plt.show()

