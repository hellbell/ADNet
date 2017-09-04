function [curr_bbox, action_history, this_actions, conv_feats, ims, actions, ah_onehot, fr_toc] = ...
    tracker_rl(curr_img, curr_bbox, my_net_conv, my_net_fc, action_history, opts)
% TRACKER_RL
% 
% Sangdoo Yun, 2017.


fr_tic = tic;

move_counter = 1;
num_action_step_max = opts.num_action_step_max;
num_show_actions = opts.num_show_actions;
imSize = opts.imgSize;
bb_step = zeros(num_action_step_max, 4);
this_actions = zeros(num_show_actions,1);
conv_feats =  zeros(3,3,512, num_action_step_max, 'single');
prev_score = -inf;

ims = zeros([112,112,3,num_action_step_max], 'single');
ah_onehot = zeros(opts.num_actions*opts.num_action_history, num_action_step_max);
actions = zeros(num_action_step_max, 1);

while (move_counter <= num_action_step_max)
    bb_step(move_counter,:) = curr_bbox;
    
    my_net_conv.mode = 'test';
    [curr_feat_conv, im] = get_conv_feature(my_net_conv, curr_img, curr_bbox, opts);
    ims(:,:,:, move_counter) = im;
    conv_feats(:,:,:, move_counter) = curr_feat_conv;
    curr_feat_conv = gpuArray(curr_feat_conv);
    my_net_fc.mode = 'test';
    labels = zeros(1);
    ah_oh = get_action_history_onehot(action_history(1:opts.num_action_history), opts);
    ah_oh = reshape(ah_oh, [1,1,numel(ah_oh),1]);
    ah_onehot(:,move_counter) = ah_oh;
    inputs = {'x10', curr_feat_conv, 'action_history', gpuArray( ah_oh)};
    my_net_fc.eval(inputs);
    my_net_pred = squeeze(gather(my_net_fc.getVar('prediction').value));
    my_net_pred_score = squeeze(gather(my_net_fc.getVar('prediction_score').value));
    curr_score = my_net_pred_score(2);
    [~, my_net_max_action] = max(my_net_pred(1:end));
    if (prev_score < -5.0 && curr_score < prev_score)
        if randn(1) < 0.5
            my_net_max_action = randi(11);
        end
    end
    actions(move_counter) = my_net_max_action;
    curr_bbox = do_action(curr_bbox, opts, my_net_max_action, imSize);
    
    % when come back again (oscillating)
    [~, ism] = ismember(round(bb_step), round(curr_bbox), 'rows');
    if sum(ism) > 0 ...
            && my_net_max_action ~= opts.stop_action        
        my_net_max_action = opts.stop_action;
    end    
    
    action_history(2:end) = action_history(1:end-1);
    action_history(1) = my_net_max_action;
    this_actions(move_counter) = 1;
    
    if my_net_max_action == opts.stop_action
        break;
    end    
    move_counter = move_counter + 1;
    prev_score = curr_score;
end
actions(move_counter+1:end) = [];
conv_feats(:,:,:,move_counter+1:end) = [];
ims(:,:,:, move_counter+1:end) = [];
ah_onehot(:, move_counter+1:end) = [];
fr_toc = toc(fr_tic);
