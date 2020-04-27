#test loading of keras retinanet
import matplotlib
matplotlib.use("MacOSX")

import os
import pytest
from keras_retinanet.preprocessing import four_channel
from deepforest import utilities
from deepforest import get_data
from matplotlib import pyplot


def test_keras_retinanet():
    import keras_retinanet
    
def test_Cython_build():
    import keras_retinanet.utils.compute_overlap  
    assert os.path.exists(keras_retinanet.utils.compute_overlap.__file__)
    
@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations(get_data("OSBS_029.xml"))    
    annotations_file = get_data("testfile_deepforest.csv")
    annotations.to_csv(annotations_file,index=False,header=False)
    
    return annotations_file

@pytest.fixture()
def classes_file(annotations):
    classes_file = utilities.create_classes(annotations)    
    return classes_file
    
def test_four_channel(annotations, classes_file):
    generator = four_channel.FourChannelGenerator(annotations, classes_file, chm_dir="tests/data/CHM/")
    inputs, targets = generator.__getitem__(0)
    assert inputs.shape == (1,800,800,4)
    
    #View   
    #fig = pyplot.figure(figsize=(30,30))
    #fig,axes = pyplot.subplots(1,2)
    #axes = axes.flatten()    
    #axes[0].imshow(inputs[0,:,:,:3])
    #axes[1].imshow(inputs[0,:,:,3])
    #data=axes[1].pcolor(inputs[0,:,:,3])
    #fig.colorbar(data)
    #pyplot.show()
    