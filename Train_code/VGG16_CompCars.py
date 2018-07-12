
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import h5py
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import pickle


def train_on_ComCars(model,out_name, image_folder = 'data/image',ismodelfile = True, fix_layer = -1):
    nb_epoch = 70
    batch_size = 32
    num_classes = 163
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # CompCars image data
    train_data_dir = image_folder
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size)

    if ismodelfile:
        base_model = load_model(model)
    else:
        base_model = model
    x = base_model.get_layer(name='block5_pool').output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    if fix_layer > -1:
        for layer in model.layers[:fix_layer]:
            layer.trainable = False

    sgd = SGD(lr=1e-3)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=4274,
    )
    model.save(out_name)
    return model


def VGG16_pool5_extract(model, name_file_path, image_folder, output_name, ismodelfile = True):
    with open(name_file_path) as name_file:
        name_list = name_file.read().splitlines()
    if ismodelfile:
        base_model = load_model(model)
    else:
        base_model = model

    X_list = []
    for name in name_list:
        image_dir = os.path.join(image_folder, name)
        x = load_img(image_dir, target_size=(img_rows, img_cols ))
        x = img_to_array(x)
        X_list.append(x)
    X_list = np.array(X_list, ndmin=4)

    x = base_model.get_layer(name='block5_pool').output
    x = Flatten()(x)
    model = Model(input=base_model.input, output=x)
    feature_arrays = model.predict(X_list, verbose=1)

    feature_arrays = np.array(feature_arrays).T
    h5f = h5py.File(output_name, 'w')
    h5f.create_dataset('VGG16_dataset', data=feature_arrays)
    h5f.close()
    return feature_arrays


def read_h5py_file(file_path):
    feature_arrays = []
    f = h5py.File(file_path)
    for _, v in f.items():
        feature_arrays = feature_arrays + list(v)

    feature_arrays = np.array(feature_arrays).T


    return feature_arrays


def load_pickle_file(filename, python=2):
    encoding = 'latin1'
    with open(filename, 'rb') as f:
        if python == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding=encoding)

    feature_matrix = []
    for _, value in data.items():
        feature_matrix.append(value)

    feature_matrix = np.array(feature_matrix[0])
    feature_matrix = np.squeeze(feature_matrix)

    return feature_matrix


def compute_score_per_query(query, test_list):
    score_list = []
    query = query
    candidate_list = test_list
    for cad in candidate_list:
        distance = np.linalg.norm(query - cad)
        score_list.append(distance)
    return score_list


# create distance file
def compute_distance_file(feature_test_file, feature_query_file, dist_file_name, caffe=False):
    if caffe == False:
        test_data = read_h5py_file(feature_test_file).T
        query_data = read_h5py_file(feature_query_file).T
    else:
        test_data = load_pickle_file(feature_test_file)
        query_data = load_pickle_file(feature_query_file)

    matrix = []
    i = 1
    for query in query_data:
        print('Processing query: ' + str(i))
        ranking_list = compute_score_per_query(query, test_data)
        matrix.append(ranking_list)
        i = i + 1
    matrix = np.array(matrix).T

    np.savetxt(dist_file_name, matrix, fmt='%10.5f')



if __name__ == '__main__':
    img_rows, img_cols = 150, 150  # Resolution of inputs
    channel = 3
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_cols,img_rows,channel))

    C_model = train_on_ComCars(base_model,image_folder='../Data/CompCars/image', out_name='../Models/VGG16/model_VGG16_CompCars.h5',ismodelfile=False)
    VGG16_pool5_extract(C_model, '../Data/VeRi/name_query.txt', '../Data/VeRi/image_query', '../Features/name_query_VGG16_CompCars.h5', ismodelfile=False)
    VGG16_pool5_extract(C_model, '../Data/VeRi/name_test.txt', '../Data/VeRi/image_test', '../Features/name_test_VGG16_CompCars.h5', ismodelfile=False)
    compute_distance_file('../Features/name_test_VGG16_CompCars.h5', '../Features/name_query_VGG16_CompCars.h5',
                          '../Evaluation/Distance_files/dist_VGG16_CompCars.txt')

