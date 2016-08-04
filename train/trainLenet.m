clear;clc;
path(path,'functions/');

%% ===============================================================
%load data
imageDim = 28;
numclasses = 10;
images = loadMNISTImages('dataset/train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('dataset/train-labels.idx1-ubyte');
labels(labels == 0) = 10;

%% initialparameters
lenet.layers = {
    struct('type','i','outputmaps',1)
    struct('type','C1','outputmaps',6,'kernelsize',5)
    struct('type','S2','outputmaps',6,'scale',2)
    struct('type','C3','outputmaps',16,'kernelsize',5)
    struct('type','S4','outputmaps',16,'scale',2)
    struct('type','C5','outputmaps',120)
    struct('type','F6','outputmaps',84)
    struct('type','Soft','num_classes',10)
};

lenet = initialNet(lenet,images);

%% train the cnn
opts.alpha = 0.1;
opts.batchsize = 25;
opts.numepochs = 20;
lenet = train(lenet,images,labels,opts);

%% test the cnn
testImages = loadMNISTImages('dataset/t10k-images.idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('dataset/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10;

lenet = cnnff(lenet,testImages);

a = lenet.layers{8}.a;
[~,preds] = max(lenet.layers{8}.a,[],1);
preds = preds';

acc = sum(preds == testLabels) / length(preds);
fprintf('Accuracy is %f\n',acc);
