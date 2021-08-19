clear; close; clc;

%set Dataset Path
salDir = '../result/';
gtDir = '/media/nercms/NERCMS/YuchengChou/VSOD_TestSet/';
Results_Save_Path = './EvalTxt/FSNet-c/';

Models = {'FSNet-New'}; %'2015-CVPR-SAG', '2014-TCSVT-SPVM', 
Datasets = {'DAVIS'};%, 'FBMS', 'SegTrack-V2', 'MCL', 'DAVSOD', 'DAVSOD-Difficult-20', 'DAVSOD-Normal-25'}; %', 'DAVSOD-Difficult-20', 'FBMS', 'SegTrack-V2', 'ViSal','YouTube-Objects', 'DAVSOD', 'DAVSOD-Normal-25' ,'MCL', 'UVSD', 'VOS'

Thresholds = 1:-1/255:0;

for  m = 1:length(Models)
    
    modelName = Models{m}

    resVideoPath = [salDir modelName '/'];
    
    videoFiles = dir(gtDir);
    
    videoNUM = length(videoFiles)-2;
    
    [video_Smeasure, video_wFmeasure, video_adpFmeasure, video_adpEmeasure, video_MAE] = deal(zeros(1,videoNUM));
    [video_Fmeasure,video_Emeasure] = deal(zeros(videoNUM,256));
    
    for videonum = 1:length(Datasets)
        videofolder = Datasets{videonum}
        filePath = [Results_Save_Path modelName '/'];
    
        if ~exist(filePath, 'dir')
            mkdir(filePath);
        end
        
        fileID = fopen([filePath modelName '_' videofolder '_result.txt'], 'w');
        
        
        seqPath = [gtDir '/' videofolder '/'];  % modified by Gepeng_Ji
        seqFiles = dir(seqPath);
        
        seqNUM = length(seqFiles)-2;
        
        [seq_Smeasure, seq_wFmeasure, seq_adpFmeasure, seq_adpEmeasure, seq_MAE] = deal(zeros(1,seqNUM));
        [seq_Fmeasure,seq_Emeasure] = deal(zeros(seqNUM,256));
        
        for seqnum = 1: seqNUM
            
            seqfolder = seqFiles(seqnum+2).name;
            
            gt_imgPath = [seqPath seqfolder '/GT/'];
            [fileNUM, gt_imgFiles, fileExt] = calculateNumber(gt_imgPath);
            resPath = [resVideoPath videofolder '/' seqfolder '/'];
            
            [Smeasure, wFmeasure, adpFmeasure, adpEmeasure, mae] = deal(zeros(1, fileNUM-2));
            [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(fileNUM-2,256));
            
            tic;
            for i = 2:fileNUM-1 %skip the first and last gt file for some of the optical-flow based method
                
                name = char(gt_imgFiles{i});
                fprintf('[Processing] Model: %s, Dataset: %s, Seq: %s (%d/%d), Name: %s (%d/%d)\n',modelName, videofolder, seqfolder, seqnum, seqNUM, name, i-1, fileNUM-2);
                
                %load gt
                gt = imread([gt_imgPath name]);
                if numel(size(gt))>2
                    gt = rgb2gray(gt);
                end
                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end
                
                %load salency
                sal  = imread([resPath name]);
                %check size
                if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                    sal = imresize(sal,size(gt));
                    imwrite(sal,[resPath name]);
                    fprintf('Error occurs in the path: %s!!!\n', [resPath name]);
                end
                
                sal = im2double(sal(:,:,1));
                
                %normalize sal to [0, 1]
                sal = reshape(mapminmax(sal(:)',0,1),size(sal));
                
                Smeasure(i-1) = StructureMeasure(sal,logical(gt));
                   
                wFmeasure(i-1) = original_WFb(sal, logical(gt));
                
                % Using the 2 times of average of sal map as the threshold.
                threshold =  2* mean(sal(:)) ;
                [~,~,adpFmeasure(i-1)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
                % Mean Absolute Error
                mae(i-1) = mean2(abs(double(logical(gt)) - sal));
                
                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold) = 1;
                adpEmeasure(i) = Enhancedmeasure(Bi_sal, gt);
                
%                 for t = 1:length(Thresholds)
%             
%                     threshold = Thresholds(t);
%                     [~, ~, threshold_Fmeasure(i-1,t)] = Fmeasure_calu(sal, double(gt), size(gt), threshold);
%                     Bi_sal = zeros(size(sal));
%                     Bi_sal(sal>threshold) = 1;
%                     threshold_Emeasure(i-1,t) = Enhancedmeasure(Bi_sal, gt);
%                 end
                
            end
            toc;
            
            seq_Smeasure(seqnum) = mean2(Smeasure);
            
            seq_wFmeasure(seqnum)= mean2(wFmeasure);
            
            seq_adpFmeasure(seqnum) = mean2(adpFmeasure);
            seq_Fmeasure(seqnum,:) = mean(threshold_Fmeasure,1);
            seq_maxF = max(seq_Fmeasure(seqnum,:));
            seq_meanF = mean(seq_Fmeasure(seqnum,:));
            
            seq_adpEmeasure(seqnum) = mean2(adpEmeasure);
            seq_Emeasure(seqnum,:) = mean(threshold_Emeasure,1);
            seq_meanE = mean(seq_Emeasure(seqnum,:));
            seq_maxE = max(seq_Emeasure(seqnum,:));
            
            
            seq_MAE(seqnum) = mean2(mae);
            
            fprintf(fileID,'(%s Dataset, %s Sequence) seq_Smeasure:%.3f;seq_wFmeasure:%.3f;seq_adpFmeasure:%.3f;seq_maxF:%.3f;seq_meanF:%.3f;seq_adpEmeasure:%.3f;seq_maxE:%.3f;seq_meanE:%.3f;seq_MAE:%.3f\n', ...
                videofolder,seqfolder,seq_Smeasure(seqnum),seq_wFmeasure(seqnum), seq_adpFmeasure(seqnum),seq_maxF,seq_meanF,seq_adpEmeasure(seqnum),seq_maxE,seq_meanE,seq_MAE(seqnum));
            fprintf('(%s Dataset, %s Sequence) seq_Smeasure:%.3f;seq_wFmeasure:%.3f;seq_adpFmeasure:%.3f;seq_maxF:%.3f;seq_meanF:%.3f;seq_adpEmeasure:%.3f;seq_maxE:%.3f;seq_meanE:%.3f;seq_MAE:%.3f\n', ...
                videofolder,seqfolder,seq_Smeasure(seqnum),seq_wFmeasure(seqnum), seq_adpFmeasure(seqnum),seq_maxF,seq_meanF,seq_adpEmeasure(seqnum),seq_maxE,seq_meanE,seq_MAE(seqnum));
            
        end
        
        video_Smeasure(videonum) = mean2(seq_Smeasure);
        video_wFmeasure(videonum) = mean2(seq_wFmeasure);
        video_adpFmeasure(videonum) = mean2(seq_adpFmeasure);
        video_adpEmeasure(videonum) = mean2(seq_adpEmeasure);
        
        video_Fmeasure(videonum,:) = mean(seq_Fmeasure,1);
        maxF = max(video_Fmeasure(videonum,:));
        meanF = mean(video_Fmeasure(videonum,:));
        
        video_Emeasure(videonum,:) = mean(seq_Emeasure,1);
        maxE = max(video_Emeasure(videonum,:));
        meanE = mean(video_Emeasure(videonum,:));
        
        video_MAE(videonum) = mean2(seq_MAE);
        % TODO: PR-Curve
%         save([resPath])
        fprintf(fileID,'(%s Dataset) seq_Smeasure:%.3f;seq_wFmeasure:%.3f;seq_adpF:%.3f;seq_maxF:%.3f;seq_meanF:%.3f;seq_adpE:%.3f;seq_maxE:%.3f;seq_meanE:%.3f;seq_MAE:%.3f\n',...
            videofolder,video_Smeasure(videonum),video_wFmeasure(videonum),video_adpFmeasure(videonum),maxF,meanF,video_adpEmeasure(videonum),maxE,meanE,video_MAE(videonum));
        fprintf('(%s Dataset) seq_Smeasure:%.3f;seq_wFmeasure:%.3f;seq_adpF:%.3f;seq_maxF:%.3f;seq_meanF:%.3f;seq_adpE:%.3f;seq_maxE:%.3f;seq_meanE:%.3f;seq_MAE:%.3f\n',...
            videofolder,video_Smeasure(videonum),video_wFmeasure(videonum),video_adpFmeasure(videonum),maxF,meanF,video_adpEmeasure(videonum),maxE,meanE,video_MAE(videonum));
    end
    
    fclose(fileID);
   
end

