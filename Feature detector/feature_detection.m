clear all; 
close all;
clc;







%% Harris detector

I = imread("Image0.jpg");
I1 = imread("Image1.jpg");
figure;
subplot(1,2,1), imshow(I);title("image ");
subplot(1,2,2), imshow(I1);title("image rotated");


% conver to gray image
I_gray = rgb2gray(I);
I1_gray = rgb2gray(I1);


points00 = detectHarrisFeatures(I_gray);
points01 = detectHarrisFeatures(I1_gray);

strongest00 = selectStrongest(points00,50);
strongest01 = selectStrongest(points01,50);
figure;
subplot(1,2,1), imshow(I_gray);title("Harris origin");
hold on;
plot(strongest00);
subplot(1,2,2), imshow(I1_gray);title("Harris origin");
hold on;
plot(strongest01);





%% FAST detector

points10 = detectFASTFeatures(I_gray);
points11 = detectFASTFeatures(I1_gray);


figure;

subplot(1,2,1), imshow(I_gray);title("FAST origin");
hold on;
plot(selectStrongest(points10,50));
subplot(1,2,2), imshow(I1_gray);title("FAST rotated");
hold on;
plot(selectStrongest(points11,50)); 




