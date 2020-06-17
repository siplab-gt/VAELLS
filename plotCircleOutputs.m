% Plot figures for concentric circle experiment

folderUse = './results/TOVAE_concen_circle_M4_A3_Nc1_batch30_rw0.01_pol11.0_poR1.0_poC1e-06_prl10.01_prR1.0_prC5e-06_g0.01/';

fileName = 'circleDataTests.mat';

psi_file = './pretrained_models/concen_circle/spreadInferenceTest.mat';
load([psi_file]);
load([folderUse fileName]);

% Plot embedding
class_0_idx = find(sample_labels == 0);
class_1_idx = find(sample_labels == 1);

figure;plot(z_np(class_0_idx,1),z_np(class_0_idx,2),'.','MarkerSize',17);
hold all;plot(z_np(class_1_idx,1),z_np(class_1_idx,2),'x','MarkerSize',12,'LineWidth',1.5);
axis off;
axis([-1.3 1.3 -1.3 1.3])
saveas(gcf,[folderUse 'ptEmbedding_vis.png']);
saveas(gcf,[folderUse 'ptEmbedding_vis.fig']);

% Plot pairs of points and paths inferred between them
for k = 1:5
figure;hold all;plot(z_np(:,1),z_np(:,2),'k.','MarkerSize',10);plot(z0_path_store(k,1),z0_path_store(k,2),'go','MarkerSize',12,'LineWidth',3);plot(z1_path_store(k,1),z1_path_store(k,2),'bx','MarkerSize',12,'LineWidth',3);
path_use = reshape(z_path_all(k,:,:),100,2);
plot(path_use(:,1),path_use(:,2),'LineWidth',3);
axis off;
saveas(gcf,[folderUse 'inferredPaths_' num2str(k) '.png']);
saveas(gcf,[folderUse 'inferredPaths_' num2str(k) '.fig']);
end


% Plot encoded points and the transport operator orbits on top
class_0_idx = find(sample_labels == 0);
class_1_idx = find(sample_labels == 1);

xtotal = [z_np(class_0_idx(1),1) z_np(class_0_idx(1),2);z_np(class_1_idx(1),1) z_np(class_1_idx(1),2)];

N = 2;
M = 4;
t = -95:2:95;
xt = zeros(N,length(t));
 figure('Position',[50 200 1600 200]);
for m_idx = 1:M
    subplot(1,4,m_idx);hold all;
    

    for kk = 1:length(xtotal)
        
        xtest = xtotal(kk,:)';
        
        xt = zeros(N,length(t));
        Psi1 = reshape(Psi_new(:,m_idx),N,N);
        for t_idx = 1:length(t)
            xt(:,t_idx) = expm(Psi1*t(t_idx))*xtest;
        end
        plot(xt(1,:),xt(2,:),'LineWidth',3.5);
        axis([-1.5 1.5 -1.5 1.5]);
%         axis equal
        axis off
    end
    plot(z_np(:,1),z_np(:,2),'k.','MarkerSize',10);


end
saveas(gcf,[folderUse 'transOptOrbits_vis.png']);
saveas(gcf,[folderUse 'transOptOrbits_vis.fig']);

Psi_mag = sqrt(sum(Psi_new.^2));
figure;bar(Psi_mag);
% hold all;plot([-1 5],[Psi_mag(1)*0.7 Psi_mag(1)*0.7],'k','LineWidth',2);
xlabel('Transport Operator Number');
ylabel('Operator Magnitude');
xlim([0.5 4.5]);
saveas(gcf,[folderUse 'circleTransOptMag.png']);
saveas(gcf,[folderUse 'circleTransOptMag.fig']);

N = 2;
M = 4;
t = -100:2:100;
xt = zeros(N,length(t));
figure;
for m_idx = 1:1
    hold all;
    

    for kk = 1:length(xtotal)
        
        xtest = xtotal(kk,:)';
        
        xt = zeros(N,length(t));
        Psi1 = reshape(Psi_new(:,m_idx),N,N);
        for t_idx = 1:length(t)
            xt(:,t_idx) = expm(Psi1*t(t_idx))*xtest;
        end
        plot(xt(1,:),xt(2,:),'LineWidth',3.5);
        axis([-1.3 1.3 -1.3 1.3]);
%         axis equal
        axis off
    end
    plot(z_np(:,1),z_np(:,2),'k.','MarkerSize',10);


end
saveas(gcf,[folderUse 'transOptOrbits_vis_1.png']);
saveas(gcf,[folderUse 'transOptOrbits_vis_1.fig']);

% Plot encoded point with sampled points on top

figure;
hold all;
for k = 1:size(anchors,1)
    z_use = reshape(z_anchor_samp(:,k,:),100,2);
    plot(z_use(:,1),z_use(:,2),'.','MarkerSize',10);
    
end
plot(anchors(:,1),anchors(:,2),'kx','MarkerSize',12,'LineWidth',3);
axis off;
saveas(gcf,[folderUse 'anchorSample.png']);
saveas(gcf,[folderUse 'anchorSample.fig']);

