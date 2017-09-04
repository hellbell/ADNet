function train_db = get_train_dbs(vid_info, opts)
% GET_VIDEO_INFOS Get video informations (image paths and ground truths)
% adopted from MDNet (Hyeonseob Nam, 2015)
% 
% Sangdoo Yun, 2017.

img = imread(vid_info.img_files(1).name);
opts.scale_factor = 1.05;
opts.imgSize = size(img);
gt_skip = opts.train.gt_skip;

if strcmp(vid_info.db_name, 'alov300')
    train_sequences = find(vid_info.gt_use==1);
else
    train_sequences = 1 : gt_skip :  vid_info.nframes ;
end
train_db = struct('img_path', cell(1, numel(train_sequences)), ...
    'bboxes', cell(1, numel(train_sequences)), ...
    'labels', cell(1, numel(train_sequences)));

for train_i = 1 : numel(train_sequences)    
    img_idx = train_sequences(train_i);
    gt_bbox = vid_info.gt(img_idx, :);
    
    pos_examples = [];
    while(size(pos_examples,1) < opts.nPos_train-1)
        pos = gen_samples('gaussian', gt_bbox, opts.nPos_train*5, opts, 0.1, 5);
        r = overlap_ratio(pos, gt_bbox);
        pos = pos(r>opts.posThre_train, :);
        if isempty(pos), continue; end
        pos = pos(randsample(end,min(end,opts.nPos_train-1-size(pos_examples,1))),:);
        pos_examples = [pos_examples;pos];
    end
    
    neg_examples = [];
    while(size(neg_examples,1)<opts.nNeg_train-1)
        neg = gen_samples('gaussian', gt_bbox, opts.nPos_train*5, opts, 2, 10);
        r = overlap_ratio(neg, gt_bbox);
        neg = neg(r<opts.posThre_train, :);
        if isempty(neg), continue; end
        neg = neg(randsample(end,min(end,opts.nNeg_train-size(neg_examples,1))),:);
        neg_examples = [neg_examples;neg];
    end    
    examples = [pos_examples; neg_examples];
    action_labels = gen_action_labels(opts.num_actions, opts, pos_examples, gt_bbox);
    
    train_db(train_i).img_path = vid_info.img_files(img_idx).name;
    train_db(train_i).bboxes = examples;
    train_db(train_i).labels = action_labels;
    train_db(train_i).score_labels = [2*ones(size(pos_examples,1),1); ones(size(neg_examples,1),1)];
end