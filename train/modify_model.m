function my_net = modify_model(my_net, opts)
% MODIFY_MODEL 
% 
% Sangdoo Yun, 2017.


for i=1:numel(my_net.layers)
    if strcmp(my_net.layers{i}.type,'conv')
        my_net.layers{i}.filtersMomentum = zeros(size(my_net.layers{i}.filters), ...
            class(my_net.layers{i}.filters)) ;
        my_net.layers{i}.biasesMomentum = zeros(size(my_net.layers{i}.biases), ...
            class(my_net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if ~isfield(my_net.layers{i}, 'filtersLearningRate')
            my_net.layers{i}.filtersLearningRate = 1 ;
        end
        if ~isfield(my_net.layers{i}, 'biasesLearningRate')
            my_net.layers{i}.biasesLearningRate = 2 ;
        end
        if ~isfield(my_net.layers{i}, 'filtersWeightDecay')
            my_net.layers{i}.filtersWeightDecay = 1 ;
        end
        if ~isfield(my_net.layers{i}, 'biasesWeightDecay')
            my_net.layers{i}.biasesWeightDecay = 0 ;
        end        
        my_net.layers{i}.filtersMomentum = gpuArray(my_net.layers{i}.filtersMomentum);
        my_net.layers{i}.biasesMomentum = gpuArray(my_net.layers{i}.biasesMomentum);        
    end
end

% mdnet.layers = mdnet.layers(1:10);
my_net.meta.inputSize = [112,112,3];
my_net.layers(end-1:end) = [];
num_actions = opts.num_actions;
my_net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'fc6', ...
                           'filters', 0.01 * randn(1,1,512,num_actions,'single'), ...
                           'biases', zeros(1, num_actions, 'single'), ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'filtersLearningRate', 10, ...
                           'biasesLearningRate', 20, ...
                           'filtersWeightDecay', 1, ...
                           'biasesWeightDecay', 0) ;
my_net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
my_net = vl_simplenn_tidy(my_net);
for i = 1 : numel(my_net.layers)
    l = my_net.layers{i} ;
    switch l.type
        case 'conv'            
            l.learningRate = [l.filtersLearningRate l.biasesLearningRate];
            l.weightDecay = [1 0];
            for j = 1 : numel(l.weights)
                l.momentum{j} = single(zeros(size(l.weights{j})));
            end
    end
    my_net.layers{i} = l;
end

my_net = vl_simplenn_tidy(my_net);