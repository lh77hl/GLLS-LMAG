clc,clear,close all


clc;
clear;
close all;
addpath('.\utils\differential operators\')
addpath('.\utils\high-order tensor-SVD Toolbox\')
addpath('.\functions\')

%% setup parameters
lambdaL =4;%
L1=20;%

saveDir1= '.\test_results\T\';
saveDir2= '.\test_results\B\';

imgpath='.\data\';

imgDir = dir([imgpath '*.bmp']);
len = length(imgDir);
tic
for i=1:len
    picname=[imgpath  imgDir(i).name];
    I=imread(picname);%
    [m,n]=size(I);
    [~, ~, ch]=size(I);
    if ch==3
        I=rgb2gray(I); 
    end
    D(:,:,i)=I;
end
tenD=double(D);
[n1,n2,n3]=size(tenD);

n_1=max(n1,n2);%n(1)
n_2=min(n1,n2);%n(2)
patch_frames=L1;% temporal slide parameter
patch_num=n3/patch_frames;

%% constrcut image tensor
for l=1:patch_num
    l
    for i=1:patch_frames
        temp(:,:,i)=tenD(:,:,patch_frames*(l-1)+i);
    end           
        lambda4 =lambdaL / sqrt(max(n_1,n_2)*patch_frames);
        % dim=size(temp);
        % A=dim(1);
        % B=dim(2);
%% The proposed model

%     [lambda1, lambda2] = structure_tensor_lambda(temp, 3);%
% %      step 2: calculate corner strength function
%     cornerStrength = (((lambda1.*lambda2)./(lambda1 + lambda2)));%
% %      step 3: obtain final weight map
%     maxValue = (max(lambda1,lambda2));%
%     priorWeight = mat2gray(cornerStrength .* maxValue);%公式17
% %      step 4: constrcut patch tensor of weight map
%     tenW = priorWeight;

opts = [];
opts.rho = 1.1;
opts.directions = [1,2,3];
opts.lambda     = 0.02;%lamuda1
opts.tol        = 1e-4;
opts.rho        = 1.1;
opts.mu         = 1e-4;%mu
opts.mu1        = 1e-5;%beta
                             
[tenB,tenT,obj,iter] = draft_GLLS_LMAG(temp, opts);


%% recover the target and background image       
 for i=1:patch_frames
     tarImg=tenT(:,:,i);
     backImg=tenB(:,:,i);
     maxv = max(max(double(I)));
     tarImg=double(tarImg);
     backImg=double(backImg);

     T = uint8( mat2gray(tarImg)*maxv );
     B = uint8( mat2gray(backImg)*maxv );

%% save the results
% imwrite(T, [saveDir1 imgDir(i+patch_frames*(l-1)).name]);     % Save target image 
% imwrite(B, [saveDir2 imgDir(i+patch_frames*(l-1)).name]); % Save background image
end 
end
toc  

