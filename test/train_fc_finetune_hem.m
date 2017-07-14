function [net_fc, hard_score_examples] = ...
    train_fc_finetune_hem(net_fc, opts, pos_data, neg_data, ...
    pos_action_labels)
% TRAIN_FC_FINETUNE_HEM finetune the network using hard negative mining 
% (many are from MDNet)
% 
% Sangdoo Yun, 2017.

USE_ACTION_HISTORY = 0;
ACTION_ORDER = [1 2 3 4 5 6 7 9] ;
SCORE_ORDER = [1 2 3 4 5 6 8 10] ;
FULL_ORDER  =[1 2 3 4 5 6 7 8 9 10];

opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.minibatch_size = 64;
opts.scale_factor = 1.05;
score_training_interval = 1;

state = [];
if isempty(state) || isempty(state.solverState)
  state.solverState = cell(1, numel(net_fc.params)) ;
  state.solverState(:) = {0} ;
end
for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
        state.solverState{i} = gpuArray(s) ;
    elseif isstruct(s)
        state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
    end
end
opts.train.solver = @solver.adam ;  % Empty array means use the default SGD solver
% [opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.train.solver)
  assert(isa(opts.train.solver, 'function_handle') && nargout(opts.train.solver) == 2,...
    'Invalid solver; expected a function handle with two outputs.') ;
  % Call without input arguments, to get default options
  opts.train.solverOpts = opts.train.solver() ;
end

% select batch
opts.nPos_online = opts.nPos_online * 2;
opts.nNeg_online = opts.nNeg_online * 2;
num_total_pos_samples = size(pos_data,4);
num_total_neg_samples = size(neg_data,4);

neg_feat_conv = gpuArray(neg_data);
pos_feat_conv = gpuArray(pos_data);

randpos = randperm(num_total_pos_samples);
randneg = randperm(num_total_neg_samples);

train_action_cnt = 0;
train_action = [];
remain = opts.minibatch_size*opts.maxiter;
while(remain>0)
    if(train_action_cnt==0)
        train_action_list = randpos';
    end
    train_action = cat(1,train_action,...
        train_action_list(train_action_cnt+1:min(end,train_action_cnt+remain)));
    train_action_cnt = min(length(train_action_list),train_action_cnt+remain);
    train_action_cnt = mod(train_action_cnt,length(train_action_list));
    remain = opts.minibatch_size*opts.maxiter-length(train_action);
end

train_neg_cnt = 0;
train_neg = [];
remain = 4*opts.minibatch_size*opts.maxiter/score_training_interval;
while(remain>0)
    if(train_neg_cnt==0)
        train_hard_list = randneg';
    end
    train_neg = cat(1,train_neg,...
        train_hard_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_hard_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_hard_list));
    remain = 4*opts.minibatch_size*opts.maxiter/score_training_interval-length(train_neg);
end

% do training
inds = find(pos_action_labels);
[pos_action_labels,~] = ind2sub(size(pos_action_labels), inds);

for i = 1 : opts.num_actions
    action_label_indices{i} = find(pos_action_labels==i);
end

num_actions_in_batch = repmat(ceil(opts.minibatch_size/opts.num_actions), [opts.num_actions, 1]);
num_actions_in_batch(opts.stop_action) = opts.minibatch_size - num_actions_in_batch(1)*(opts.num_actions-1);
for i = 1 : opts.num_actions
    train_action_cnt = 0;
    train_action = [];
    remain = num_actions_in_batch(i)*opts.maxiter;
    action_label_index = action_label_indices{i};
    randpos = action_label_index(randperm(numel(action_label_index)));    
    if isempty(randpos)
        randpos = randperm(num_total_pos_samples);
        randpos = randpos';
    end
    while(remain>0)        
        if(train_action_cnt==0)
            train_action_list = randpos;
        end        
        train_action = cat(1,train_action,...
            train_action_list(train_action_cnt+1:min(end,train_action_cnt+remain)));
        train_action_cnt = min(length(train_action_list),train_action_cnt+remain);
        train_action_cnt = mod(train_action_cnt,length(train_action_list));
        remain = num_actions_in_batch(i)*opts.maxiter-length(train_action);
    end
    train_actions{i} = train_action;
end

l1 = 0;
l2 = 0;

for iter = 1 : opts.maxiter    
    
    % training (action)
    pos_examples = [];
    for i = 1 : opts.num_actions
        train_action = train_actions{i};
        pos_examples = [pos_examples; train_action((iter-1)*num_actions_in_batch(i)+1: (iter)*num_actions_in_batch(i))];        
    end
    pos_feat = pos_feat_conv(:,:,:, pos_examples);
    easy_score_labels = 2*ones(size(pos_examples));
    action_labels = pos_action_labels(pos_examples);    
    
    inputs = {'x10', pos_feat, 'label', gpuArray(action_labels)};
    outputs = {'objective', 1};
        
    net_fc.mode = 'test';
    net_fc.setExecutionOrder(ACTION_ORDER);
    net_fc.reset;
    net_fc.eval(inputs, outputs) ;
    state = accumulate_gradients_dagnn(net_fc, state, opts.train, opts.minibatch_size) ;
    l1 = gather(net_fc.vars(11).value);
    
    if mod(iter, score_training_interval) == 0
        % hard score example mining
        net_fc.mode = 'test';
        net_fc.setExecutionOrder(SCORE_ORDER)   ;     
        num_batches = 4 ;
        scores = zeros(num_batches*opts.minibatch_size, 1);
        hard_start = (iter/score_training_interval-1) * num_batches * opts.minibatch_size;
        for batchIdx = 1 : num_batches
            b_st = (batchIdx-1)*opts.minibatch_size;
            curr_batch = train_neg(hard_start+b_st+1:hard_start+b_st+opts.minibatch_size);
            curr_feat = neg_feat_conv(:,:,:, curr_batch);
            curr_score_labels = 1*ones(size(curr_batch));
            
            inputs = {'x10', curr_feat, 'label_score', gpuArray(curr_score_labels)};
            net_fc.reset;
            net_fc.eval(inputs) ;
            pred = gather(squeeze(net_fc.vars(net_fc.getVarIndex('prediction_score')).value));
            scores(b_st+1:b_st+opts.minibatch_size) = pred(2,:);
        end
        [~,ord] = sort(scores,'descend');
        hard_score_examples = train_neg(hard_start+ord(1:opts.minibatch_size));
        hard_score_feat = neg_feat_conv(:,:,:, hard_score_examples);
        hard_score_labels = 1*ones(size(hard_score_examples));

        % training (score)
        curr_feat = cat(4, hard_score_feat, pos_feat);
        curr_labels = [hard_score_labels; easy_score_labels];
        net_fc.mode = 'test';        
        
        inputs = {'x10', curr_feat, 'label_score', gpuArray(curr_labels)};
        outputs = {'objective_score', 1};
        net_fc.reset;
        net_fc.eval(inputs, outputs) ;    
        [state] = accumulate_gradients_dagnn(net_fc, state, opts.train, opts.minibatch_size*2);
        
        pred = squeeze(gather(net_fc.vars(9).value));
        lab = pred(1,:) < pred(2,:);
        lab = lab + 1;
%         pred = pred(2,:);
        l2 = gather(net_fc.vars(13).value);
    end
%     fprintf('%d: %f, %f\n',iter, l1,  l2);
end
net_fc.setExecutionOrder(FULL_ORDER);

