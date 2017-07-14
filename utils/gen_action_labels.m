function [action_labels, overs] = gen_action_labels(num_actions, opts, bb_samples, gt_bbox)
% GEN_ACTION_LABELS generate action labels for training the network
% 
% Sangdoo Yun, 2017.
num_samples = size(bb_samples, 1);

action_labels = zeros(num_actions , num_samples);
m = opts.action_move;

for j =  1 : size( bb_samples, 1)
    bbox = bb_samples(j, :);
    % action
    bbox(1) = bbox(1) + 0.5*bbox(3);
    bbox(2) = bbox(2) + 0.5*bbox(4);
    
    deltas = [m.x*bbox(3) m.y*bbox(4) m.w*bbox(3) m.h*bbox(4)];
    deltas = max(deltas, 1);
    ar = bbox(3)/bbox(4);
    if bbox(3) > bbox(4)
        deltas(4) = deltas(3) / ar;
    else
        deltas(3) = deltas(4) * ar;
    end
    deltas = repmat(deltas, [num_actions, 1]);
    action_deltas = m.deltas .* deltas;
    
    action_boxes = repmat(bbox, [num_actions, 1]);
    action_boxes = action_boxes + action_deltas;
    action_boxes(:, 1) = action_boxes(:, 1) - 0.5 * action_boxes(:, 3);
    action_boxes(:, 2) = action_boxes(:, 2) - 0.5 * action_boxes(:, 4);
    
    overs = overlap_ratio(action_boxes, gt_bbox);
    [max_value, max_action] = max(overs(1:end-2)); % translation overlap 
    if overs(opts.stop_action) > opts.stopIou
        max_action = opts.stop_action;
    end
    if max_value == overs(opts.stop_action)
        [~, max_action] = max(overs(1:end)); % (trans + scale) action 
    end
    action = zeros(num_actions,1);
    action(max_action) = 1;
    action_labels(:, j) = action;
end