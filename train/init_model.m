function [net, dummy_nets, dummy_score_nets] = init_model(dst_path, opts)
% INIT_MODEL Initialize ADNet using DagNN wrapper
% 
% Sangdoo Yun, 2017.

src_model = opts.vgg_m_path;

if exist(dst_path,'file')
    load(dst_path);    
    if isfield(net, 'net');
        net = net.net;
    end    
else
    % load conv layers
    load(src_model);
    % initialize networks
    simplenn_net = init_model_networks(layers);
    % modify networks to our model
    simplenn_net = modify_model(simplenn_net, opts);
    simplenn_net.layers = simplenn_net.layers(1:end-2);
    % convert to dagNN
    dagnn_net = dagnn.DagNN();
    dagnn_net = dagnn_net.fromSimpleNN(simplenn_net, 'CanonicalNames', true);
    net = dagnn_net;
    
    % add two fc6 layers (fc6_1, fc6_2)
    % FC6_1 layer: action output
    block1 = dagnn.Conv('size', [1 1 512 opts.num_actions], 'hasBias', true);
    net.addLayer('fc6_1', block1, {'x16'}, {'prediction'}, ...
        {'fc6_1f', 'fc6_1b'});
    net.params(net.getParamIndex('fc6_1f')).value =  0.01 * randn(1,1,512,opts.num_actions,'single');
    net.params(net.getParamIndex('fc6_1f')).learningRate = 10;
    net.params(net.getParamIndex('fc6_1b')).value =  zeros(1, opts.num_actions, 'single');
    net.params(net.getParamIndex('fc6_1b')).learningRate = 20;
    
    % FC6_2 layer: binary classification output
    block2 = dagnn.Conv('size', [1 1 512 2], 'hasBias', true);
    net.addLayer('fc6_2', block2, {'x16'}, {'prediction_score'}, ...
        {'fc6_2f', 'fc6_2b'});
    net.params(net.getParamIndex('fc6_2f')).value =  0.01 * randn(1,1,512,2,'single');
    net.params(net.getParamIndex('fc6_2f')).learningRate = 10;
    net.params(net.getParamIndex('fc6_2b')).value =  zeros(1, 2, 'single');
    net.params(net.getParamIndex('fc6_2b')).learningRate = 20;
    
    % Loss 1 layer: softmax of action output (11 dim)
    softmaxlossBlock1 = dagnn.Loss('loss', 'softmaxlog');
    net.addLayer('loss',softmaxlossBlock1,{'prediction', 'label'},{'objective'});
    
    % Loss 2 layer: softmax of object/background (2 dim)
    softmaxlossBlock2 = dagnn.Loss('loss', 'softmaxlog');
    net.addLayer('loss_score',softmaxlossBlock2,{'prediction_score', 'label_score'},{'objective_score'});
    
end

% ------ Multi-Domain --------
block = dagnn.Conv('size', [1 1 512 opts.num_actions], 'hasBias', true);
dummy_nets = {};
for i = 1 : opts.num_videos
    dmnet = dagnn.DagNN();
    layer_name = ['v' num2str(i)];    
    dmnet.addLayer(layer_name, block, {'x16'}, {'prediction'}, {'fc6_1f', 'fc6_1b'});
    f_idx = net.getParamIndex('fc6_1f');
    b_idx = net.getParamIndex('fc6_1b');
    dmnet.params(dmnet.getParamIndex('fc6_1f')).value = 0.01 * randn(1,1,512,opts.num_actions,'single');
    dmnet.params(dmnet.getParamIndex('fc6_1f')).learningRate = 10;
    dmnet.params(dmnet.getParamIndex('fc6_1b')).value = zeros(1, opts.num_actions, 'single');
    dmnet.params(dmnet.getParamIndex('fc6_1b')).learningRate = 20;
    dummy_nets{i} = dmnet;
end

% scoring network
block = dagnn.Conv('size', [1 1 512 2], 'hasBias', true);
dummy_score_nets = {};
for i = 1 : opts.num_videos
    dmnet = dagnn.DagNN();
    layer_name = ['vs' num2str(i)];    
    dmnet.addLayer(layer_name, block, {'x16'}, {'prediction_score'}, {'fc6_2f', 'fc6_2b'});
    f_idx = net.getParamIndex('fc6_2f');
    b_idx = net.getParamIndex('fc6_2b');
    dmnet.params(dmnet.getParamIndex('fc6_2f')).value = 0.01 * randn(1,1,512,2,'single');
    dmnet.params(dmnet.getParamIndex('fc6_2f')).learningRate = 10;
    dmnet.params(dmnet.getParamIndex('fc6_2b')).value = zeros(1, 2, 'single');
    dmnet.params(dmnet.getParamIndex('fc6_2b')).learningRate = 20;
    dummy_score_nets{i} = dmnet;
end

save(dst_path, 'net');
