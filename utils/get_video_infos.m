function [video_info] =  get_video_infos(bench_name, video_path, video_name)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

switch bench_name
    case {'vot13','vot14','vot15'}
        % path to VOT dataset
        video_info.gt = [];
        video_info.img_files = [];
        video_info.name = video_name;        
        benchmarkSeqHome = video_path;
        video_info.db_name = bench_name;
        
        % img path
        imgDir = fullfile(benchmarkSeqHome, video_name);
        if(~exist(imgDir,'dir'))
            error('%s does not exist!!',imgDir);
        end
        img_files = dir(fullfile(imgDir,'*.jpg'));
        for i = 1 : numel(img_files)
            video_info.img_files(i).name = fullfile(video_path, video_name, img_files(i).name);
        end
        
        % gt path
        gtPath = fullfile(benchmarkSeqHome, video_name, 'groundtruth.txt');
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        % parse gt
        gt = importdata(gtPath);
        if size(gt,2) >= 6
            x = gt(:,1:2:end);
            y = gt(:,2:2:end);
            gt = [min(x,[],2), min(y,[],2), max(x,[],2) - min(x,[],2), max(y,[],2) - min(y,[],2)];
        end
        video_info.gt = gt;
        
        video_info.nframes = min(length(video_info.img_files), size(video_info.gt,1));
        video_info.img_files = video_info.img_files(1:video_info.nframes);
        video_info.gt = video_info.gt(1:video_info.nframes,:);                        
end
