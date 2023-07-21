%NOTE that one of the j has to be commented

j= imread('IMG_20191126_163725_BURST001.jpg'); %retrieve picture(far distance 80KMH sign)
%j= imread('IMG_20191126_163725_BURST040.jpg'); %retrieve picture(close distance 80KMH sign)

rotated = imrotate(j,270,'bilinear'); %rotate picture

cropped = imcrop(rotated,[420,591,1500,2700]);% x : 405 - 2300    |    y : 591 - 3400  
                                              %([x min, y min, x max(whidth), y max(height)]) 
                                              
grey = rgb2gray(cropped); % convert to grayscale


bw = im2bw(grey); % Convert to black & white with autmatic threshold chooser

preComp = imcomplement(bw); %imcomplement the picture

finalFilters = imclose(preComp, strel('disk', 10)); %Performing the morphological closing

[labels,numlabels]=bwlabel(finalFilters);

coloured = label2rgb(labels); %Fill in RGB colour in every blob
imshow(coloured);

gen_subplots(rotated, cropped, grey, bw, finalFilters, labels, coloured); % Subplot

eighty_sign_stats = regionprops (labels,'all'); %get the specifications of the object
eighty_sign_stats(1) %acquisition specified for merely object one

formFactor = 4*pi*eighty_sign_stats(1).Area/((eighty_sign_stats(1).Perimeter)^2) %form factor calculation for precise classification

eightsign_intensity = eighty_sign_stats(1).Area / 4000; %Distance calculation
max_distance = 100;
sign_distance = max_distance - eightsign_intensity;

if sign_distance > 0
    fprintf("The distance of the object is: %f cm", sign_distance); %Distance indication
else
    fprintf("Unknown distance");
end