import tensorflow as tf
from Task4Model import t4
from customloss import cseloss, LossCse673

from Dataloader import DataGeneratorCars
from scipy.io import loadmat
import itertools

import numpy as np
from cubsArch1_Cars2 import cubsArch1

if __name__ == "__main__":
    path_to_train_images = '../carsData/update/car_ims'
    # labelsTest = {'00001.jpg': 1, '00002.jpg': 2, '00003.jpg': 3, '00004.jpg': 4, '00005.jpg': 5}

    path_to_labels = '../carsData/update/cars_annos.mat'
    # path_to_classes = './devkit/cars_meta.mat'
    # print(path_to_classes)

    mat_train = loadmat(path_to_labels)
    # print(mat_train)
    mat_train = dict(itertools.islice(mat_train.items(), len(mat_train)))
    labels = {}
    labelstest = {}

    for example in mat_train['annotations'][0]:
        label = example[5][0]
        image = example[0][0].split('/')[1]
        if label < 99:
            labels[image] = label
        else:
            labelstest[image] = label

    labels = dict(itertools.islice(labels.items(), len(labels)))
    # print("labels", type(labels))

    X_valid = {}
    X_train = {}
    # from collections import defaultdict
    inv_map = {}
    for k in labels:
        value = list(labels[k])[0]
        if value in inv_map:
            inv_map[value].append(k)
        else:
            inv_map[value] = [k]
    train_perc = 0.8

    # for i in inv_map:
    #     values = list(inv_map[i])
    #     X_train[i] = values[:np.int(np.ceil(len(inv_map[i])*train_perc))]
    #     X_valid[i] = values[np.int(np.ceil(len(inv_map[i])*train_perc)):]
    for i in inv_map:
        values = list(inv_map[i])
        for r in range(len(values)):
            if r <= np.int(np.ceil(len(inv_map[i]) * train_perc)):
                # if i not in X_train:
                X_train[values[r]] = i
                # else:
                #     X_train[i].append(values[r])
            else:
                # if i not in X_valid:
                X_valid[values[r]] = i
                # else:
                # X_valid[i].append(values[r])
    ##Old Model
    preTrained = tf.keras.models.load_model("cars98_cubs")
    ##New Model
    model = t4(preTrained)
    # model = cubsArch1(98)


    ##alpha = 1 - 10
    ##beta = 40 - 60

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
    #               metrics=["accuracy"])
    customloss = cseloss(alpha=1, beta=40,  batch_size=200)
    # ##SGD with momentum
    sgd_momentum = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9
    )
    model.compile(optimizer=sgd_momentum, loss=customloss,
                  metrics=["accuracy"])
    ######## CallBacks
    reduceonplateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        verbose=1.0,
        mode="auto",
        min_delta=0.05,
        cooldown=0,
        min_lr=0.0
    )

    ####### Data Generators

    trainGenerator = DataGeneratorCars(
        labels=X_train, rescale=1. / 255, path_to_train=path_to_train_images, val=False, flag=1,
        batch_size=200, target_size=(64, 64), channels=3, n_classes=98)

    ##Does batch size have to be exact divisor of training samplesx
    validGenerator = DataGeneratorCars(
        labels=X_valid, rescale=1. / 255, path_to_train=path_to_train_images, val=True, flag=0,
        batch_size=200, target_size=(64, 64), channels=3, n_classes=98)

    testGenerator = DataGeneratorCars(
        labels=X_valid, rescale=1. / 255, path_to_train=path_to_train_images, val=True, flag=0,
        batch_size=10, target_size=(64, 64), channels=3, n_classes=98)

    #####Model Fit
    model.fit_generator(generator=trainGenerator,
                        validation_data=validGenerator,
                        use_multiprocessing=True,
                        epochs=100,
                        # initial_epoch=50,
                        callbacks=[reduceonplateau],
                        workers=10)

    # model.save("cars98_cubs")
    loss, acc = model.evaluate_generator(testGenerator, steps=3, verbose=1)
    print(acc)
    print(loss)






