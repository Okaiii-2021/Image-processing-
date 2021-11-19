clc; 
close all;
clear all;


% Read the image
I = imread("digits.jpg");

figure;
imshow(I);
title("Original Image");

% convert to gray image
I_gray = rgb2gray(I);


% template image
T = I_gray(1:50, 100:171);

figure;
imshow(T);
title("template");

h = vision.TemplateMatcher;

Loc = step(h,I_gray, T);


figure;
imshow(I_gray);title("result-matching location");
hold on;
plot(Loc(1,1),Loc(1,2), 'r*',  "linewidth", 5);






