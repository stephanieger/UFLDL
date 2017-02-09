function patches = sampleMNIST(file)

images = loadMNISTImages(file);
numpatches = 10000;
x = randi(size(images,2),1,numpatches);
patches = zeros(size(images,1),numpatches);
for i = 1:numpatches
    patches(:,i) = images(:,x(i));
end

end