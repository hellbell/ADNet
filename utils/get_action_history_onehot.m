function ah_onehot = get_action_history_onehot(action_history, opts)
% GET_ACTION_HISTORY_ONEHOT returns action history as one-hot form
% 
% Sangdoo Yun, 2017.
ah_onehot = [];
for i = 1 : numel(action_history)    
    onehot = zeros(1, opts.num_actions);
    if action_history(i) > 0 && action_history(i) <= opts.num_actions
        onehot(action_history(i)) = 1;
    end
    ah_onehot = [ah_onehot, onehot];     
end


