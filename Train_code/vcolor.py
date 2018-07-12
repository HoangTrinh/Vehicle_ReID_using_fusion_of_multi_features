import caffe
import pickle
import numpy as np
import h5py

def get_this_batch(image_list, batch_index, batch_size):
    start_index = batch_index * batch_size
    next_batch_size = batch_size
    image_list_size = len(image_list)
    # batches might not be evenly divided
    if(start_index + batch_size > image_list_size):
        reamaining_size_at_last_index = image_list_size - start_index
        next_batch_size = reamaining_size_at_last_index
    batch_index_indices = range(start_index, start_index+next_batch_size,1)
    return image_list[batch_index_indices]


def extract_vcolor(input_images_file, model_def, pretrained_model, extract_from_layer, outname):
    batch_size = 1
    gpu_id = 0
    ext_file = open(input_images_file, 'r')
    image_paths_list = [line.strip() for line in ext_file]
    ext_file.close()

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list]

    net = caffe.Net(model_def, pretrained_model, caffe.TEST)

    # set up transformer - creates transformer object
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # transpose image from HxWxC to CxHxW
    transformer.set_transpose('data', (2, 0, 1))

    # swap image channels from RGB to BGR
    transformer.set_channel_swap('data', (2, 1, 0))

    # set raw_scale = 255 to multiply with the values loaded with caffe.io.load_image
    transformer.set_raw_scale('data', 227)

    total_batch_nums = len(images_loaded_by_caffe)
    features_all_images = []
    images_loaded_by_caffe = np.array(images_loaded_by_caffe)

    # loop through all the batches
    for j in range(total_batch_nums):
        image_batch_to_process = get_this_batch(images_loaded_by_caffe, j, batch_size)
        num_images_being_processed = len(image_batch_to_process)
        data_blob_index = range(num_images_being_processed)

        # note that each batch is passed through a transformer
        # before passing to data layer
        net.blobs['data'].data[data_blob_index] = [transformer.preprocess('data', img) for img in
                                                   image_batch_to_process]

        # BEWARE: blobs arrays are overwritten
        res = net.forward()

        # actual batch feature extraction
        features_for_this_batch = net.blobs[extract_from_layer].data[data_blob_index].copy()
        features_all_images.extend(features_for_this_batch)
        print(j)

    pkl_object = {"filename": image_paths_list, "features": features_all_images}
    output_pkl_file_name = outname

    output = open(output_pkl_file_name, 'wb')
    pickle.dump(pkl_object, output, 2)
    output.close()

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

    extract_vcolor("../Data/VeRi/name_train.txt", "../Models/Vcolor/vcolor_deploy02.prototxt", "../Models/Vcolor/vcolor_train02_RGB_t01_iter_250000.caffemodel", "fc7",
                   '../Features/vcolor_train.pkl')

    extract_vcolor("../Data/VeRi/name_test.txt", "../Models/Vcolor/vcolor_deploy02.prototxt", "../Models/Vcolor/vcolor_train02_RGB_t01_iter_250000.caffemodel", "fc7",
                   '../Features/vcolor_test.pkl')

    extract_vcolor("../Data/VeRi/name_query.txt", "../Models/Vcolor/vcolor_deploy02.prototxt", "../Models/Vcolor/vcolor_train02_RGB_t01_iter_250000.caffemodel", "fc7",
                   '../Features/vcolor_query.pkl')
    compute_distance_file('../Features/vcolor_test.pkl', '../Features/vcolor_query.pkl', '../Evaluation/Distance_files/dist_vcolor.txt')