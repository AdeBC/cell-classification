# sh: ls [Path] > fileNameofCell.txt
# needed
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

def loadData():
    data={'cell':[]}
    directory= '/home/xueyu/ch379/cell_classification/src/'

    for i in range(438):
        image= cv2.imread(directory+'/data/cell_'+str(i)+'.png')
        data['cell'].append(image)
    
    f=open(directory+'label.txt')
    label_read= f.read().split(',')
    f.close()
    
    l= list(map(int, label_read))
    data['label']= l
    
    return data


def buildModel(x_train):
    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
        filters=32, 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='valid', 
        activation='relu')
    )

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='valid', 
        activation='relu')
    )

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model


def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('roc_curve.png')


if __name__=="__main__":

    data= loadData()
    x_train, x_test, y_train, y_test = train_test_split(np.array(data['cell']),
    np.array(data['label']),
    test_size=0.2,
    random_state=3)

    model= buildModel(x_train)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
                )
                
    model.summary()

    # train model
    model.fit(x_train, y_train, epochs=200)
    res = model.evaluate(x_test, y_test)
    
    y_pred = model.predict_classes(tf.cast(x_test, tf.float32))
    fper, tper, thresholds = roc_curve(y_test, y_pred)
    
    plot_roc_curve(fper, tper)
    AUC= auc(fper, tper)
    
    r=open('classifier.result', 'w')
    r.write('y_test\n'+','.join(list(map(str, y_test)))+'\n\n')
    r.write('y_prediction\n'+','.join(list(map(str, y_pred))))
    r.close()

    model.save('classifier_ch379.h5')
    print('loss: {}, accuracy: {}'.format(res[0], res[1]))
    print('AUC: ', AUC)