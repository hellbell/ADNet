function net = adnet_train_RL(net, train_videos, opts)
% ADNET_TRAIN_RL Train the ADNet with reinforcement learning
% 
% Sangdoo Yun, 2017.

video_names = train_videos.video_names;
video_paths = train_videos.video_paths;
bench_names = train_videos.bench_names;

load('./models/net_dummy.mat');

net.addLayer('concat_',dagnn.Concat, {'x16', 'action_history'}, 'x_16_ah');
net.layers(end-3:end-1) = net.layers(end-4:end-2); % make room for concat
net.layers(17) = net.layers(end);
net.layers(17).name = 'concat';
net.layers(18).inputs = {'x_16_ah'};
net.layers(19).inputs = {'x_16_ah'};
net.removeLayer('concat_');
net.rebuild();
nvec = opts.num_actions * opts.num_action_history;
net.params(11).value = cat(3, net.params(11).value, zeros(1,1,nvec,11));
net.params(13).value = cat(3, net.params(13).value, zeros(1,1,nvec,2));

% Convert softmaxlog -> policy gradient loss
net.layers(net.getLayerIndex('loss')).block.loss = 'softmaxlog_pg';

% Initialize momentum
state.momentum = num2cell(zeros(1, numel(net.params)));
state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);

% Set SGD Excution order 
exec_order = net.getLayerExecutionOrder;
exec_order_action = exec_order;
exec_order_action(end-1:2:end) = [];

% =========================================================================
% RUN TRACKING ON TRAINING VIDEOS
% =========================================================================
opts.minibatch_size = 32;

tic_train = tic;
for epoch = 1 : opts.numEpoch
    vid_idxs = randperm(numel(video_names));
    for vid_idx = vid_idxs
        % -----------------------------------------------------------------
        % Load current training video info
        % -----------------------------------------------------------------
        video_name = video_names{vid_idx};
        video_path = video_paths{vid_idx};
        bench_name = bench_names{vid_idx};
        vid_info = get_video_infos(bench_name, video_path, video_name);
        curr_img = imread(fullfile(vid_info.img_files(1).name));
        imSize = size(curr_img);
        opts.imgSize = imSize;
        num_show_actions = 20;
        
        % -----------------------------------------------------------------
        % Load FC6 Layer's Parameters for Multi-domain Learning
        % -----------------------------------------------------------------
        net.move('gpu');
        dummy_net = dummy_nets{vid_idx};
        dummy_net.move('gpu');        
        layer_name = ['v' num2str(vid_idx)];
        fc_layer = dummy_net.getLayer(layer_name);
        net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:) = ...
            dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value;
        net.params(net.getParamIndex(fc_layer.params{2})).value = dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value;
        
        dummy_score_net = dummy_score_nets{vid_idx};
        dummy_score_net.move('gpu');
        layer_name = ['vs' num2str(vid_idx)];
        fc_layer_score = dummy_score_net.getLayer(layer_name);
        net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:) = ...
            dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value;
        net.params(net.getParamIndex(fc_layer_score.params{2})).value = dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value;
        
        
        [net, net_conv, net_fc] = split_dagNN(net);
        net_conv.move('gpu');
        net_fc.move('gpu');
        obj_idx = net_fc.getVarIndex('objective');        
        pred_idx = net_fc.getVarIndex('prediction');        
        conv_feat_idx = net_conv.getVarIndex('x10');
        net_fc.vars(obj_idx).precious = 1;
        net_fc.vars(pred_idx).precious = 1;
        net_conv.vars(conv_feat_idx).precious = 1;
        
        % -----------------------------------------------------------------
        % Sample k video clips (between the given ground truths)
        % -----------------------------------------------------------------        
        RL_steps = opts.train.RL_steps;
        vid_clip_starts = 1:size(vid_info.gt, 1) - RL_steps;
        vid_clip_ends = vid_clip_starts + RL_steps;
        vid_clip_ends(end) = size(vid_info.gt, 1);
        if vid_clip_starts(end) == vid_clip_ends(end)
            vid_clip_starts(end) = [];
            vid_clip_ends(end) = [];
        end
        randp = randperm(numel(vid_clip_starts));
        vid_clip_starts = vid_clip_starts(randp);
        vid_clip_ends = vid_clip_ends(randp);
        num_train_clips = min(opts.train.rl_num_batches, numel(vid_clip_starts));
        
        % -----------------------------------------------------------------
        % Play sampled video clips & calculate target value (z)
        % -----------------------------------------------------------------
        curr_loss = 0;
        ah_pos = []; ah_neg = [];
        imgs_pos = []; action_labels_pos = [];
        imgs_neg = []; action_labels_neg = [];
        for clipIdx = 1:num_train_clips
            frameStart = vid_clip_starts(clipIdx);
            frameEnd = vid_clip_ends(clipIdx);
            curr_bbox = vid_info.gt(frameStart,:);
            action_history = zeros(num_show_actions, 1);
%             this_actions = zeros(num_show_actions, 1);
            %
            imgs = [];
            action_labels = [];
            ah_onehots = [];
            for frameIdx = frameStart+1:frameEnd
                curr_img = imread(fullfile(vid_info.img_files(frameIdx).name));
                if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end
            
                curr_gt_bbox = vid_info.gt(frameIdx,:);
                [curr_bbox, action_history, ~, ~, ims, actions, ah_onehot, ~] = ...
                    tracker_rl(curr_img, curr_bbox, net_conv, net_fc, action_history, opts);
                ah_onehots = cat(2, ah_onehots, ah_onehot);
                imgs = cat(4, imgs, ims);
                action_labels = cat(1, action_labels, actions);
            end
            % target value (z)
            if overlap_ratio(curr_gt_bbox, curr_bbox) > 0.7
                ah_pos = cat(2, ah_pos, ah_onehots);
                imgs_pos = cat(4, imgs_pos, imgs);
                action_labels_pos = cat(1, action_labels_pos, action_labels);
            else
                ah_neg = cat(2, ah_neg, ah_onehots);
                imgs_neg = cat(4, imgs_neg, imgs);
                action_labels_neg = cat(1, action_labels_neg, action_labels);
            end                      
        end
        
        % -----------------------------------------------------------------
        % RL Training using Policy Gradient
        % -----------------------------------------------------------------
        num_pos = size(action_labels_pos, 1);
        num_neg = size(action_labels_neg, 1);        
        train_pos_cnt = 0;
        train_pos = [];
        train_neg_cnt = 0;
        train_neg = [];
        batch_size = opts.minibatch_size;
        if num_pos > batch_size/2
            % Random permutation batches (Pos)
            remain = opts.minibatch_size * num_train_clips;
            while(remain>0)
                if(train_pos_cnt==0)
                    train_pos_list = randperm(num_pos);
                    train_pos_list = train_pos_list';
                end
                train_pos = cat(1,train_pos,...
                    train_pos_list(train_pos_cnt+1:min(numel(train_pos_list),train_pos_cnt+remain)));
                train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
                train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
                remain = opts.minibatch_size * num_train_clips-length(train_pos);
            end
        end        
        if num_neg > batch_size/2
            % Random permutation batches (Neg)
            remain = opts.minibatch_size * num_train_clips;
            
            while(remain>0)
                if(train_neg_cnt==0)
                    train_neg_list = randperm(num_neg);
                    train_neg_list = train_neg_list';
                end
                train_neg = cat(1,train_neg,...
                    train_neg_list(train_neg_cnt+1:min(numel(train_neg_list),train_neg_cnt+remain)));
                train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
                train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
                remain = opts.minibatch_size * num_train_clips-length(train_neg);
            end
        end
        
        % Training       
        for batchIdx = 1 : num_train_clips            
            if ~isempty(train_pos)
                pos_examples = train_pos((batchIdx-1)*batch_size+1:batchIdx*batch_size);
                imgs = imgs_pos(:,:,:, pos_examples);
                action_labels = action_labels_pos(pos_examples);   
                ahs = ah_pos(:,pos_examples);
                ahs = reshape(ahs, [1,1, size(ahs,1), size(ahs,2)]);
                inputs = {'input', gpuArray(imgs), 'label', gpuArray(action_labels), 'action_history', gpuArray(ahs)};
                outputs = {'objective', 1};
                net.setExecutionOrder(exec_order_action);
                target = 1;
                net.layers(net.getLayerIndex('loss')).block.opts = {'target', target / batch_size};
                net.eval(inputs, outputs);
                [state] = accumulate_gradients_dagnn(state, net, opts.train, batch_size);
                curr_loss = curr_loss + gather(net.getVar('objective').value) / batch_size;
            end
            if ~isempty(train_neg)
                neg_examples = train_neg((batchIdx-1)*batch_size+1:batchIdx*batch_size);
                imgs = imgs_neg(:,:,:, neg_examples);
                action_labels = action_labels_neg(neg_examples);
                ahs = ah_neg(:,neg_examples);
                ahs = reshape(ahs, [1,1, size(ahs,1), size(ahs,2)]);
                inputs = {'input', gpuArray(imgs), 'label', gpuArray(action_labels), 'action_history', gpuArray(ahs)};
                outputs = {'objective', 1};
                net.setExecutionOrder(exec_order_action);
                target = -1;
                net.layers(net.getLayerIndex('loss')).block.opts = {'target', target / batch_size};
                net.eval(inputs, outputs);
                [state] = accumulate_gradients_dagnn(state, net, opts.train, batch_size);
                curr_loss = curr_loss + gather(net.getVar('objective').value) / batch_size;
            end
        end

        
        % Print objective (loss)
        curr_loss = curr_loss / num_train_clips;
        fprintf('time: %04.2f, epoch: %02d/%02d, vid: %05d/%05d \n', ...
            toc(tic_train), epoch, opts.numEpoch, vid_idx, numel(video_names));
        
        % -----------------------------------------------------------------
        % Save current parameters
        % -----------------------------------------------------------------
        dummy_net.params(dummy_net.getParamIndex(fc_layer.params{1})).value = net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:);
        dummy_net.params(dummy_net.getParamIndex(fc_layer.params{2})).value = net.params(net.getParamIndex(fc_layer.params{2})).value;
        dummy_net.move('cpu');
        dummy_nets{vid_idx} = dummy_net;        
        dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{1})).value = net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:);
        dummy_score_net.params(dummy_score_net.getParamIndex(fc_layer_score.params{2})).value = net.params(net.getParamIndex(fc_layer_score.params{2})).value;
        dummy_score_net.move('cpu');
        dummy_score_nets{vid_idx} = dummy_score_net;        
    end
end

% -------------------------------------------------------------------------
% FINALIZE TRAINING
% -------------------------------------------------------------------------
net.move('cpu');
net.params(net.getParamIndex(fc_layer.params{1})).value(:,:,1:512,:) = 0.01 * randn(1,1,512,opts.num_actions,'single');
net.params(net.getParamIndex(fc_layer.params{2})).value = zeros(1, opts.num_actions, 'single');
net.params(net.getParamIndex(fc_layer_score.params{1})).value(:,:,1:512,:) = 0.01 * randn(1,1,512,2,'single');
net.params(net.getParamIndex(fc_layer_score.params{2})).value = zeros(1, 2, 'single');

softmaxlossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('loss_score',softmaxlossBlock,{'prediction_score', 'label_score'},{'objective_score'});

