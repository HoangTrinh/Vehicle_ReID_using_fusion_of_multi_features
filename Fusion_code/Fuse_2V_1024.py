import keras
import pickle
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from keras.layers import Dense, BatchNormalization, concatenate
from keras.models import Model, load_model


def model_creation(feature_1, feature_2, n_classes):
    input1 = keras.layers.Input(shape=(feature_1.shape[1],))
    b1 = Dense(feature_1.shape[1], activation='relu')(input1)
    b1 = BatchNormalization()(b1)
    input2 = keras.layers.Input(shape=(feature_2.shape[1],))
    b2 = Dense(feature_2.shape[1], activation='relu')(input2)
    b2 = BatchNormalization()(b2)

    added = concatenate(([b1, b2]))
    added = keras.layers.Dense(1024, activation='relu')(added)
    predictions = keras.layers.Dense(n_classes, activation='softmax')(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=predictions)
    model.summary()
    return model


def train_on_VeRi(vgg16_train_file_1, vgg16_train_file_2, out_name, xml_name_file_path='train_label.xml'):
    n_classes = 776
    vgg16_feature = read_h5py_file(vgg16_train_file_1)
    vcolor_feature = read_h5py_file(vgg16_train_file_2)

    xmlp = ET.XMLParser(encoding="utf-8")
    f = ET.parse(xml_name_file_path, parser=xmlp)
    root = f.getroot()

    labels = []
    for child in root.iter('Item'):
        labels.append(child.attrib['vehicleID'])

    model = model_creation(vgg16_feature, vcolor_feature, n_classes)

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=n_classes)

    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit([vgg16_feature, vcolor_feature], one_hot_labels, batch_size=1024, epochs=200)

    model.save(out_name)
    return model


def Fuse_extract(base_model_path, vgg16_feature_path_1, vgg16_feature_path_2, output_name, ismodelfile=True):
    vgg16_feature = read_h5py_file(vgg16_feature_path_1)
    vcolor_feature = read_h5py_file(vgg16_feature_path_2)

    if ismodelfile:
        base_model = load_model(base_model_path)
    else:
        base_model = base_model_path

    x = base_model.get_layer(index=-2).output
    model = Model(input=base_model.input, output=x)

    feature_arrays = model.predict([vgg16_feature, vcolor_feature], verbose=1)

    feature_arrays = np.array(feature_arrays).T
    h5f = h5py.File(output_name, 'w')
    h5f.create_dataset('Fuse_dataset', data=feature_arrays)
    h5f.close()


def read_h5py_file(file_path):
    feature_arrays = []
    f = h5py.File(file_path)
    for _, v in f.items():
        feature_arrays = feature_arrays + list(v)

    feature_arrays = np.array(feature_arrays).T

    f.close()
    return feature_arrays


def load_pickle_file(filename, python = 2):
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
    model = train_on_VeRi('../Features/name_train_VGG16_VeRi.h5', '../Features/name_train_VGG16_CompCars.h5', '../Models/Fusion/model_double_2V_1024.h5',
                          xml_name_file_path='../Data/VeRi/train_label.xml')

    Fuse_extract(model, '../Features/name_query_VGG16_VeRi.h5', '../Features/name_query_VGG16_CompCars.h5', '../Features/query_double_2V_1024.h5',
                 ismodelfile=False)
    Fuse_extract(model, '../Features/name_test_VGG16_VeRi.h5', '../Features/name_test_VGG16_CompCars.h5', '../Features/test_double_2V_1024.h5',
                 ismodelfile=False)

    compute_distance_file('../Features/test_double_2V_1024.h5', '../Features/query_double_2V_1024.h5', '../Evaluation/Distance_files/dist_double_2V_1024.txt')

