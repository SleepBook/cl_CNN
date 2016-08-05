#Train Module

***This module which is implemented using Matlab is used to generate the Neural Net configure file which can be passed to the Execution Module***

###Brief Instruction
Current there is a demo which is used to train the Net Config for Lenet-5.

To run the demo, you execute the `trainLenet.m`,`NetExtractor.m`and `inputdata_converter.m` sequentially. This will generate a `test.cnet` file in the output folder, which is a description of the Lenet acccording to the format which is used by the Execution Module. The detail of this format can be referred [here](./CNN_FormatSpec.txt). The last script will generate the date for testing purpose, which is stored in `test.cdat` in the output folder.

###The Function Referred in Scripts

`loadMNISTImages`/`loadMNISTLabels`: automatically extrace the binary files from the MNIST database

`initialNet`: initialize the link values

`cnnff`: feed-forward calculation

`cnnbp`: back-propragation

`applyGrads`: use the gradient to adjust the link weights

###Training Parameter Adjustment
The parameter for training can be changed inside the `functions/train.m` file. The representation of some variables:

`alpha` : learning rate

`batchsize`: The number of samples in each training batch

`numberpochs`: total training iterations

###Develop Guide
If you want to used these scripts to generate your custom NN, just specify the Network Configure by changing `lenet.struct` structure in the `trainLenet` file, by modifying/adding/removing the layers and specifying there names(C means Convolutional Layer, S means Pooling and F means full connected), you get whatever you like. The other procedure is the same, just run the three scripts again and your Net Configure file will be generated under the `output` directory.