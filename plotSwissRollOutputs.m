folderUse = './results/TOVAE_2Restart_pretrain0_swiss2D_M1_A4_Nc1_batch30_rw0.01_pol11.0_poR1.0_poC1e-06_prl10.01_prR1.0_prC5e-05_g0.01_lr0.0001_nst20pst20/';

fileName = 'swissRollDataTests.mat';

psi_file = './pretrained_models/swiss_roll/spreadInferenceTest.mat';
load([psi_file]);
load([folderUse fileName]);

% Plot embedding
figure;plot(z_np(:,1),z_np(:,2),'.','MarkerSize',17);
axis off;
axis([-1.7 2 -1.7 2]);
saveas(gcf,[folderUse 'ptEmbedding_vis.png']);
saveas(gcf,[folderUse 'ptEmbedding_vis.fig']);

% Plot pairs of points and paths inferred between them
for k = 2:3
figure;hold all;plot(z_np(:,1),z_np(:,2),'.','MarkerSize',18,'Color',[56 41 41]/255);plot(z0_path_store(k,1),z0_path_store(k,2),'o','MarkerSize',20,'LineWidth',4,'Color',[22 219 233]/255);plot(z1_path_store(k,1),z1_path_store(k,2),'x','MarkerSize',20,'LineWidth',4,'Color',[223 83 92]/255);
path_use = reshape(z_path_all(k,:,:),100,2);
plot(path_use(:,1),path_use(:,2),'LineWidth',5.5,'Color',[119 184 81]/255);
axis off;
saveas(gcf,[folderUse 'inferredPaths_' num2str(k) '_swissRoll.png']);
saveas(gcf,[folderUse 'inferredPaths_' num2str(k) '_swissRoll.fig']);
end

% Plot encoded points and the transport operator orbits on top
class_0_idx = find(sample_labels == 0);


xtotal = [z_np(class_0_idx(1),1) z_np(class_0_idx(1),2)];

N = 2;
M = 1;
t = -80:2:280;
xt = zeros(N,length(t));
figure;
for m_idx = 1:M
    hold all;
    
    plot(z_np(:,1),z_np(:,2),'.','MarkerSize',21,'Color',[56 41 41]/255);
    for kk = 1:size(xtotal,1)
        
        xtest = xtotal(kk,:)';
        
        xt = zeros(N,length(t));
        Psi1 = reshape(Psi_new(:,m_idx),N,N);
        for t_idx = 1:length(t)
            xt(:,t_idx) = expm(Psi1*t(t_idx))*xtest;
        end
        plot(xt(1,:),xt(2,:),'LineWidth',3.5,'Color',[223 83 92]/255);
        
         axis off
    end
    


end
saveas(gcf,[folderUse 'transOptOrbits_vis_swissRoll.png']);
saveas(gcf,[folderUse 'transOptOrbits_vis_swissRoll.fig']);

% Plot encoded point with sampled points on top
colorVals = [119 184 81;22 219 233;12 48 156;223 83 92]/255;
figure;
hold all;
for k = 1:size(anchors,1)
    z_use = reshape(z_anchor_samp(:,k,:),100,2);
    plot(z_use(:,1),z_use(:,2),'.','MarkerSize',20,'Color',colorVals(k,:));
    
end
plot(anchors(:,1),anchors(:,2),'x','MarkerSize',25,'LineWidth',4,'Color',[56 41 41]/255);
axis off;
saveas(gcf,[folderUse 'anchorSample_swissRoll.png']);
saveas(gcf,[folderUse 'anchorSample_swissRoll.fig']);

