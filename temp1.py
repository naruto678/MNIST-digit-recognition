from keras.datasets import mnist
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()
import numpy as np
def make_model():
    from keras import models
    from keras import layers
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

##data preprocessing 
def preprocess(train_data):
    train_data=train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)
    train_data=train_data.astype('float32')/255
    return train_data

train_data=preprocess(train_data)
test_data=preprocess(test_data)
partial_train_data=train_data[:30000]
val_train_data=train_data[30000:]
#preprocess the labels
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
mnist_model=make_model()
partial_train_labels=train_labels[:30000]
val_train_labels=train_labels[30000:]
dic=mnist_model.fit(partial_train_data,partial_train_labels,batch_size=512,epochs=3,validation_data=(val_train_data,val_train_labels))
# plotting the model
import matplotlib.pyplot as plt
epochs=range(3)
plt.plot(epochs,dic.history['acc'],'bo',label='training accuracy')
plt.plot(epochs,dic.history['val_acc'],'b',label='validation_accuracy')
plt.plot(epochs,dic.history['loss'],'ro',label='training_loss')
plt.plot(epochs,dic.history['val_loss'],'r',label='validation_loss')
plt.legend()
plt.show()

loss,acc=mnist_model.evaluate(test_data,test_labels)

# thus we see that the model performs amazingly the accuracy comes out to be 98.27 percent
mnist_model.save('mnist_model',mnist_model)

    