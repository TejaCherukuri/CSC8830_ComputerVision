% Load image sets
setDir = fullfile(toolboxdir('vision'), 'visiondata', 'imageSets');
imgSets = imageSet(setDir, 'recursive');

% Partition the dataset
trainingSets = partition(imgSets, 0.8, 'randomize');

% Create the bag of features
bag = bagOfFeatures(trainingSets, 'Verbose', false);

% Extract features and labels from the training set
numImages = sum([trainingSets.Count]);
features = zeros(numImages, bag.VocabularySize);
labels = cell(numImages, 1);
counter = 1;
for i = 1:numel(trainingSets)
    for j = 1:trainingSets(i).Count
        img = read(trainingSets(i), j);
        features(counter, :) = encode(bag, img);
        labels{counter} = trainingSets(i).Description;
        counter = counter + 1;
    end
end

% Train a classifier (e.g., SVM)
classifier = fitcecoc(features, labels);

% Initialize variables for testing
testingFeatures = [];
testingLabels = {};

% Extract features and labels from the testing set (first two images from each image set)
for i = 1:numel(imgSets)
    for j = 1:2 % Use only the first two images for testing
        img = read(imgSets(i), j);
        testingFeatures = [testingFeatures; encode(bag, img)];
        testingLabels = [testingLabels; imgSets(i).Description];
    end
end

% Predict labels for testing features
predictedLabels = predict(classifier, testingFeatures);

% Calculate accuracy
accuracy = sum(strcmp(predictedLabels, testingLabels)) / numel(testingLabels);
disp(['Accuracy: ', num2str(accuracy)]);
