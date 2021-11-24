How to test?
In the survey they introduced a Natural-Color dataset for testing
colorization performance

1. Deep Colorization
First work to attempt using CNNs for image colorization.

Loss function: Least-squares error
5 fully connected linear layers
ReLU activations

During testing, features are extracted at three levels.

Low-level sequential gray values;
Mid-level are DAISY features
High-level semantic labeling.

Finally joint-bilateral filtering is performed
Takes 256x256 image as input and has 3 hidden layers

---

2. Colorful Colorization
Input: Grayscale image 256x256
Output: A-B channels of *Lab* colorspace 224x224

Stacks convolutional layers with 8 total blocks.
Each block consists of two or three convolutional layers followed by ReLU and BatchNorm

Striding is used to decrease the size of the image

3. Deep Depth Colorization
