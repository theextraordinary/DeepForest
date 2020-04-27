#test loading of keras retinanet
import os
import pytest
from keras_retinanet.preprocessing import four_channel
from deepforest import utilities
from deepforest import get_data

def test_keras_retinanet():
    import keras_retinanet
    
def test_Cython_build():
    import keras_retinanet.utils.compute_overlap  
    assert os.path.exists(keras_retinanet.utils.compute_overlap.__file__)
    
@pytest.fixture()
def annotations():
    annotations = utilities.xml_to_annotations(get_data("OSBS_029.xml"))
    #Point at the png version for tfrecords
    annotations.image_path = annotations.image_path.str.replace(".tif",".png")
    
    annotations_file = get_data("testfile_deepforest.csv")
    annotations.to_csv(annotations_file,index=False,header=False)
    
    return annotations_file

@pytest.fixture()
def classes_file(annotations):
    classes_file = utilities.create_classes(annotations)    
    return classes_file
    
def test_four_channel(annotations, classes_file):
    generator = four_channel.FourChannelGenerator(annotations, classes_file)
    inputs, targets = generator.__getitem__(0)
    assert inputs.shape == (1,800,800,4)
    