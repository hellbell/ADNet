function [bbox_next] = do_action(bbox, opts, act, imSize)
% DO_ACTION perform a given action 
% 
% Sangdoo Yun, 2017.
m = opts.action_move;

% action
bbox(1) = bbox(1) + 0.5 * bbox(3);
bbox(2) = bbox(2) + 0.5 * bbox(4);

deltas = [m.x*bbox(3) m.y*bbox(4) m.w*bbox(3) m.h*bbox(4)];
deltas = max(deltas, 1);
ar = bbox(3)/bbox(4);
if bbox(3) > bbox(4)
    deltas(4) = deltas(3) / ar;
else
    deltas(3) = deltas(4) * ar;
end
action_delta = m.deltas(act,:) .* deltas;
bbox_next = bbox + action_delta;
bbox_next(:, 1) = bbox_next(:, 1) - 0.5 * bbox_next(:, 3);
bbox_next(:, 2) = bbox_next(:, 2) - 0.5 * bbox_next(:, 4);
bbox_next(:, 1) = max(bbox_next(:,1), 1);
bbox_next(:, 1) = min(bbox_next(:,1), imSize(2) - bbox_next(:,3));
bbox_next(:, 2) = max(bbox_next(:,2), 1);
bbox_next(:, 2) = min(bbox_next(:,2), imSize(1) - bbox_next(:,4));
bbox_next(:, 3) = max(5, min(imSize(2), bbox_next(:, 3)));
bbox_next(:, 4) = max(5, min(imSize(1), bbox_next(:, 4)));


