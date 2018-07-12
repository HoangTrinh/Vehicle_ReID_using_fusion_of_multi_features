import os
import pickle

import h5py
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array


def VGG16_pool5_extract(model, name_file_path, image_folder, output_name, ismodelfile=True):
    with open(name_file_path) as name_file:
        name_list = name_file.read().splitlines()
    if ismodelfile:
        base_model = load_model(model)
    else:
        base_model = model

    X_list = []
    for name in name_list:
        image_dir = os.path.join(image_folder, name)
        x = load_img(image_dir, target_size=(img_rows, img_cols))
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

    f.close()
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
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))

    VGG16_pool5_extract(base_model, '../Data/VeRi/name_query.txt', '../Data/VeRi/image_query', '../Features/name_query_VGG16_ImageNet.h5', ismodelfile=False)
    VGG16_pool5_extract(base_model, '../Data/VeRi/name_test.txt', '../Data/VeRi/image_test', '../Features/name_test_VGG16__ImageNet.h5', ismodelfile=False)
    compute_distance_file('../Features/name_test_VGG16__ImageNet.h5', '../Features/name_query_VGG16_ImageNet.h5', '../Evaluation/Distance_files/dist_VGG16_ImageNet')
