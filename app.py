import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import os
from glob import glob
import json
from json import JSONEncoder
import numpy
from sklearn.neighbors import NearestNeighbors
import streamlit as st

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open('sample.json') as json_file:
    data = json.load(json_file)

resnet=models.resnet50(pretrained=True)
layer = resnet._modules.get('avgpool')
#grab all images in the lfw folder
import os
from glob import glob
path="./lfw"

result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))]
resnet.eval

#d={}

preprocess=transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
                               ])

def get_vector(image):
    # Create a PyTorch tensor with the transformed image
    t_img = preprocess(image)
    my_embedding = torch.zeros(2048)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():                               # <-- no_grad context
        resnet(t_img.unsqueeze(0))                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding

#for image in result:
  #d[image]=get_vector(Image.open(image).convert('RGB')).numpy()

image=st.file_uploader(label="upload your own file",type="jpg")
if image is None:
    st.write("upload an image")
else:
    input=get_vector(Image.open(image).convert('RGB')).numpy()
    featurelist=[]
    for img in data:
        featurelist.append(data[img])
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute',metric='euclidean').fit(featurelist)
    distances, indices = neighbors.kneighbors(input.reshape(1,-1))
    simular=[]
    for i in range(10):
        simular.append(result[indices[0][i]])
    st.image(simular,caption=simular)