

save_folder = './results/TOVAE_vAN0.0001_vAT0.0001_1start_natDigits_pre30000_CA1_M4_z6_A8_batch1_rw1.0_pol11.0_poR1.0_poC1e-06_prl10.01_prR1.0_prC1e-06_g0.01_lr0.0001_nst20pst60_samples/';
stepTrain=34000;
for m = 1:4
    %     v = VideoWriter(['./videos/transOptPath_mnist_trainSet_startNum' num2str(n) 'transOpt' num2str(m) '.avi']);
    
    load([save_folder 'transOptOrbitTest_natDigit_startDigit_step' num2str(stepTrain) '_1.mat']);
    % load(['./samples_mnist_autoencoder_trans_trainSet/transOptOrbitTest_mnist_trainSet_startDigit' num2str(n) '.mat']);
    M = size(imgOut,1);
    numStep = size(imgOut,2);
    imgSize = size(imgOut,3);
    
    c_dim = size(imgOut,5);
    figure('Position',[30 30 1000 1000]);
    %             figure('Position',[1 1 1000 400]);
    stepUse = 1:2:numStep;
    imgAll = zeros(10*imgSize,length(stepUse)*imgSize);
    for n = 1:10
        load([save_folder 'transOptOrbitTest_natDigit_startDigit_step' num2str(stepTrain) '_' num2str(n) '.mat']);
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


load([save_folder 'transOptSampleTest_natDigit_startDigit_step' num2str(stepTrain) '.mat']);
numEx = size(imgOrig,1);

M = size(imgOut,2);
numStep = size(imgOut,3);
imgSize = size(imgOut,4);

c_dim = 1;
img_size = 28;
imgPlot = zeros(3*img_size,5*img_size);
for ii = 1:numEx
    
    
    img_count = 1;
    for m = 1:5
        for n = 1:3
            imgPlot((n-1)*img_size+1:n*img_size,(m-1)*img_size+1:m*img_size) = reshape(img_samp(ii,img_count,:,:),img_size,img_size);
            img_count = img_count+1;
        end
    end
    imgPlot(img_size+1:img_size*2,2*img_size+1:3*img_size) = reshape(imgOrig(ii,:,:),img_size,img_size);
    figure;
    imagesc(imgPlot)
    axis off
    colormap('gray');
    caxis([0 1])
    title(['Transport Operator ' num2str(m)]);
    saveas(gcf,[save_folder 'transformSampImg_' num2str(stepTrain) '_class' num2str(ii) '_vaells.png']);
    fprintf('transOpt %d\n', m);
end
