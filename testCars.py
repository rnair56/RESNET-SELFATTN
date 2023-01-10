from re import L
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models, optimizers, activations
from tensorflow.keras.losses import categorical_crossentropy
import os
import tensorflow as tf
from Dataloader import DataGeneratorCars
from scipy.io import loadmat
import cubsArch1
from cubsArch1 import cubsArch1
from cubsArch1_Cars2 import cubsArch1
import itertools
from resnet import ResNet18

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

    # print("x_train", X_train)
    # print("x_valid", X_valid)

    # inv_map[k] =
    # print("keys", inv_map.keys())
    # print(inv_map[1])
    model = cubsArch1(98)
    # model = ResNet18(98)

    # model.build(input_shape=(100, 64, 64, 3))
    ##model compilation
    use_saved_model = False

    # checkpoint_filepath = 'resnetcubs_cars_checkpoint/'

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=["accuracy"])
    #
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=False,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True)

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

    # print(len(labels))
    # X_train = dict(list(labels.items())[:4050])
    # print(np.unique(np.array(list(X_train.values()))))
    # X_valid = dict(list(labels.items())[4050:])
    # print(np.unique(np.array(list(X_valid.values()))))
    # print(np.histogram(X_valid))

    print("Training labels size = ", len(X_train))
    print("Validation labels size = ", len(X_valid))
    print("Test labels size = ", len(labelstest))

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

    # validDataGenerator = validationGener.flow_from_directory(
    #     validaDir,
    #     target_size=(150, 150),
    #     class_mode='binary',
    #     batch_size=20
    # )

    # history = model.fit_generator(trainGeenrator, steps_per_epoch=100, epochs=1,
    #                               validation_data=validDataGenerator,
    #                               validation_steps=50)

    # history = model.fit_generator(trainGeenrator, steps_per_epoch=2000, epochs=10)

    # history = model.fit(trainGeenrator, steps_per_epoch=2000, epochs=10)
    # history
    # early_stoping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.005,
    #     patience=50,
    #     verbose=0,
    #     mode="auto",
    #     baseline=None,
    #     restore_best_weights=False,
    # )
    #
    # model.fit_generator(generator=trainGenerator,
    #                     validation_data=validGenerator,
    #                     use_multiprocessing=True,
    #                     epochs=100,
    #                     # initial_epoch=50,
    #                     callbacks=[reduceonplateau, early_stoping],
    #                     workers=10)
    # model.save("cars98_cubs")
    model = tf.keras.models.load_model("cars98_cubs")
    # model = tf.train.Checkpoint.restore(save_path=checkpoint_filepath).expect_partial()
    loss, acc = model.evaluate_generator(validGenerator, steps=3, verbose=0)

    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    loss, acc = model.evaluate_generator(testGenerator, steps=3, verbose=1)
    print(acc)
    print(loss)