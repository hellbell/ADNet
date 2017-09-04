function [net, net_conv, net_fc] = split_dagNN(net)
% SPLIT_DAGNN
% 
% Sangdoo Yun, 2017.

net.move('cpu');
net_fc = copy(net);
net_conv = copy(net);

layer_names = {};
for ii = 1 : 10
    layer_names{ii} = net_fc.layers(ii).name;
end
net_fc.removeLayer(layer_names);
net_fc.rebuild();

layer_names = {};
for ii = 11 : numel(net.layers)
    layer_names{ii-10} = net_conv.layers(ii).name;
end
net_conv.removeLayer(layer_names);
net_conv.rebuild();

net.move('gpu');

net_fc.rebuild();

