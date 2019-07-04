import numpy as np
import caffe

class PredictorCaffe:
    def __init__(self, model_file, pretrained_file, size):
        self.net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        self.size = size
        
    def forward(self, tensor, data="data"):
        self.net.blobs[data].data[...] = tensor
        self.net.forward()
        
    def blob_by_name(self, blobname):
        return self.net.blobs[blobname]

    def list_blob_name(self):
        # for each layer, show the output shape
        for layer_name, blob in self.net.blobs.items():
            print(layer_name + '\t' + str(blob.data.shape))