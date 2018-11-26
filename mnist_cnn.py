from optparse import OptionParser
from bigdl.examples.keras.keras_utils import *

import keras.backend
if keras.backend.image_dim_ordering() == "th":
    input_shape = (3, 32, 32)
else:
    input_shape = (32, 32, 3)


def get_cifar(sc, data_type, location="/tmp/mnist"):
    from keras.datasets import cifar10
    from bigdl.dataset.transformer import normalizer
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    if(data_type == "train"):
        print("Debugando 1")
        Y_train = np_utils.to_categorical(y_train, 10)
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        X_train = X_train.reshape((X_train.shape[0], 1) + input_shape)
        X_train = sc.parallelize(X_train)
        Y_train = sc.parallelize(Y_train + 1)  
        record = images.zip(Y_train)
        return record
    elif(data_type == "test):
        print("Debugando 2")
        Y_test = np_utils.to_categorical(y_test, 10)
        print('X_test shape:', X_train.shape)
        print(X_test.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        X_test = X_test.reshape((X_test.shape[0], 1) + input_shape)
        X_test = sc.parallelize(X_test)
        Y_test = sc.parallelize(Y_test + 1)  
        record = images.zip(Y_test)
        return record


def build_keras_model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="32")
    parser.add_option("-m", "--max_epoch", type=int, dest="max_epoch", default="200")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/cifar10")
    (options, args) = parser.parse_args(sys.argv)

    keras_model = build_keras_model()
    json_path = "/tmp/cifar10.json"
    save_keras_definition(keras_model, json_path)

    from bigdl.util.common import *
    from bigdl.nn.layer import *
    from bigdl.optim.optimizer import *
    from bigdl.nn.criterion import *

    # Load the JSON file to a BigDL model
    bigdl_model = Model.load_keras(json_path=json_path)

    sc = get_spark_context(conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    train_data = get_cifar(sc, "train", options.dataPath)
    test_data = get_cifar(sc, "test", options.dataPath)

    optimizer = Optimizer(
        model=bigdl_model,
        training_rdd=train_data,
        criterion=ClassNLLCriterion(logProbAsInput=False),
        optim_method=RMSprop(),
        end_trigger=MaxEpoch(options.max_epoch),
        batch_size=options.batchSize)
    optimizer.set_validation(
        batch_size=options.batchSize,
        val_rdd=test_data,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )
optimizer.optimize()
