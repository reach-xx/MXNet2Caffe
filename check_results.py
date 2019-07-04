from predictor_caffe import PredictorCaffe
from predictor_mxnet import PredictorMxNet
import numpy as np
from sklearn import preprocessing

def compare_diff_sum(tensor1, tensor2):
    pass

def compare_cosin_dist(tensor1, tensor2):
    pass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compare_models(prefix_mxnet, prefix_caffe, size):
    netmx = PredictorMxNet(prefix_mxnet, 0, size)    
    model_file = prefix_caffe + ".prototxt"
    pretrained_file = prefix_caffe + ".caffemodel"
    netcaffe = PredictorCaffe(model_file, pretrained_file, size)
    tensor = np.ones(size, dtype=np.float32)
    print(tensor.shape)
    out_mx = netmx.forward(tensor)
    output = out_mx[0].asnumpy()
    print(output.shape)
    print(output[0, 0:20])

    caffe_tensor = np.ones(size, dtype=np.float32)
    caffe_tensor = (caffe_tensor - 127.5) * 0.0078125
    netcaffe.forward(caffe_tensor)
    out_caffe = netcaffe.blob_by_name("fc1")
    print(out_caffe.data[0, 0:20])

    netcaffe.list_blob_name()
    dist = np.sum(output-out_caffe.data)
    print("distance: ", dist)
    #print softmax(out_caffe.data)
    #out_caffe = netcaffe.blob_by_name("fc2")
    #print(out_caffe.data)
    #print softmax(out_caffe.data)     
    print("done")
    
if __name__ == "__main__":
    prefix_mxnet = "model-r100-ii/model"
    prefix_caffe = "model_caffe/r100"
    size = (1, 3, 112, 112)
    compare_models(prefix_mxnet, prefix_caffe, size)
