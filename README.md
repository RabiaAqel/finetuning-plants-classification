# finetuning-plants-classification

Fine-Tuning on ResNet50 pre-trained on PlantClef2016 Dataset.

<p>
Using Keras high API with
TensorFlow backend. Freezing first 15 layers of
ResNet50 model pretrained on ImageNet dataset.
Dropout with rate 0.5 added before softmax
Outoput layer.
</p>


<u><h4>Dataset</h4></u>
<p>
<b>PlantClef2015</b>
1000 species of plants represented by images of the whole plant or different parts of the plants.
Train set split 80:30 for Cross-Validaiton. Each image is identified based on taxonomic ClassId.
Unique species ClassId mapped to indexes in range [0,999] representing the class id for model output.
Mapping of ImageId (Image name) to generated classes mapping saved as a numpy array file.
I rescalled the images to (224,224,3) seperatley. Possible to specify resizing
using a Keras library.
  Test on selected set of the offered test dataset. Only 4633 images corresponding to one of the 1000 species.
Source: http://www.imageclef.org/lifeclef/2016/plant
</p>


<u><h4>Metrics</h4></u>
<p>
Loss: 
Sparse Categorical Cross-Enrtropy<br>
Accuracy: 
Mean accuracy rate on all predictions
<br>
  </p>
<u><h4>Optimizer</h4></u>
<p>Adam with Learning rate 0.0001</p>

<hr>
Train set Accuracy 0.9875
Test set Accuracy 0.586
Top-3 test Accuracy 0.7335
Top-5 test Accuracy 0.785








