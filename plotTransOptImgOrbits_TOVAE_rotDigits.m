
save_folder = './results/TOVAE_vAN0.0001_vAT0.0001_1start_rotDigits_pre-1_CA1_M1_z10_A10_batch1_rw1.0_pol11.0_poR1.0_poC1e-06_prl10.01_prR1.0_prC1e-06_g0.01_lr0.0001_nst20pst60_samples/';
stepTrain=5000;
for m = 1:1
    
    load([save_folder 'transOptOrbitTest_rotDigit_startDigit_step' num2str(stepTrain) '_1.mat']);
    M = size(imgOut,1);
    numStep = size(imgOut,2);
    imgSize = size(imgOut,3);
    
    c_dim = size(imgOut,5);
    figure('Position',[30 30 1000 1000]);
    stepUse = 1:2:numStep;
    imgAll = zeros(10*imgSize,length(stepUse)*imgSize);
    for n = 1:10
        load([save_folder 'transOptOrbitTest_rotDigit_startDigit_step' num2str(stepTrain) '_' num2str(n) '.mat']);
        count = 1;
        for k = stepUse
            imgAll((n-1)*imgSize+1:n*imgSize,(count-1)*imgSize+1:count*imgSize) = reshape(imgOut(m,k,:,:,:),imgSize,imgSize,c_dim);
            
            count = count+1;
            test = 1;
        end
    end
    imagesc(imgAll)
    axis off
    colormap('gray');
    caxis([0 1])
    title(['Transport Operator ' num2str(m)]);
    saveas(gcf,[save_folder 'transformImg_' num2str(stepTrain) '_TO' num2str(m) '.png']);
    fprintf('transOpt %d\n', m);
end


load([save_folder 'transOptOrbitTest_rotDigit_randDigit_step' num2str(stepTrain) '.mat']);
numEx = size(imgOut,1);
numPlots = floor(numEx/20);
M = size(imgOut,2);
numStep = size(imgOut,3);
imgSize = size(imgOut,4);

c_dim = 1;

 load([save_folder 'transOptSampleTest_rotDigit_startDigit_step' num2str(stepTrain) '.mat']);

exNumUse = [4 60 8 48 42 83 9 61 34 58];
stepUse = 5:4:numStep-4;
imgAll_ex = zeros(5*imgSize,length(stepUse)/2*imgSize);
for n = 1:5
    count = 1;
    for k = stepUse
    imgAll_ex((n-1)*imgSize+1:n*imgSize,(count-1)*imgSize+1:count*imgSize) = reshape(imgOut(exNumUse(n),1,k,:,:,:),imgSize,imgSize,c_dim);
    count = count+1;
    end
end

figure;
imagesc(imgAll_ex);
axis off;colormap('gray');

stepUse = 5:4:numStep-4;
imgAll_ex = zeros(5*imgSize,length(stepUse)/2*imgSize);
for n = 6:10
    count = 1;
    for k = stepUse
    imgAll_ex((n-6)*imgSize+1:(n-5)*imgSize,(count-1)*imgSize+1:count*imgSize) = reshape(imgOut(exNumUse(n),1,k,:,:,:),imgSize,imgSize,c_dim);
    count = count+1;
    end
end

figure;
imagesc(imgAll_ex);
axis off;colormap('gray');

numEx = size(imgOrig,1);

M = size(imgOut,2);
numStep = size(imgOut,3);
imgSize = size(imgOut,4);

c_dim = 1;

imgPlot = zeros(3*32,5*32);
for ii = 1:numEx
    
    
    img_count = 1;
    for m = 1:5
        for n = 1:3
            imgPlot((n-1)*32+1:n*32,(m-1)*32+1:m*32) = reshape(img_samp(ii,img_count,:,:),32,32);
            img_count = img_count+1;
        end
    end
    imgPlot(32+1:32*2,2*32+1:3*32) = reshape(imgOrig(ii,:,:),32,32);
    figure;
    imagesc(imgPlot)
    axis off
    colormap('gray');
    caxis([0 1])
    saveas(gcf,[save_folder 'transformSampImg_' num2str(stepTrain) '_class' num2str(ii) '.png']);
    fprintf('transOpt %d\n', m);
end
