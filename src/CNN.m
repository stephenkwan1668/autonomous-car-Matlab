% Returns the 18 layer CNN, Little Alex
% input size is other than 256x256 the readFunctionTrain function has to be
% adjusted as well
% categories example: {'cat', 'hat'}
function [cnn]=littleAlex(trainingFolder, categories, inputSize, maxEpochs)

    imds = imageDatastore(fullfile(trainingFolder, categories), 'LabelSource', 'foldernames'); 
    % create image data store with own dataset
    imds.ReadFcn = @readFunctionTrain; 
    % resize images to 256x256
    
    conv1 = convolution2dLayer(11, 96,'Padding',2,'BiasLearnRateFactor',2, 'Stride', 4);
    % 96 filters of 11x11 with padding [2 2 2 2] and stride [4 4]
    conv1.Weights = single(randn([11 11 3 96])*0.0001);
    % initialize random weights for conv1
    fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2, 'Name', 'fc1');
    fc2 = fullyConnectedLayer(length(categories),'BiasLearnRateFactor',2, 'Name', 'fc2');
    
    % create layers
    layers = [
        imageInputLayer([inputSize inputSize 3]);
        conv1;
        reluLayer();
        crossChannelNormalizationLayer(5);
        maxPooling2dLayer(3, 'Stride', 2);
        groupedConvolution2dLayer(5, 128, 2, 'Padding', 2); % 2 groups of 128 filters, size: 5x5, padding [2 2 2 2]
        reluLayer();
        crossChannelNormalizationLayer(5); % test
        maxPooling2dLayer(3, 'Stride', 2);
        convolution2dLayer(3, 384, 'Padding', 1); % 384 filters of 3x3 padding [1 1 1 1]
        reluLayer();
        maxPooling2dLayer(3, 'Stride', 2);
        dropoutLayer(0.5);
        fc1;
        reluLayer();
        fc2;
        softmaxLayer();
        classificationLayer();
        ];
    opts = trainingOptions('sgdm', 'InitialLearnRate', 0.0001, 'VerboseFrequency', 10 ,'MaxEpochs', maxEpochs, 'Shuffle', 'every-epoch','MiniBatchSize', 32, 'Plots', 'training-progress');
    cnn = trainNetwork(imds, layers, opts);