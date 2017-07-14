function [my_net, my_net_conv, my_net_fc] = split_dagNN(my_net)
% SPLIT_DAGNN split network into `conv part' and 'fc part'
% 
% Sangdoo Yun, 2017.

my_net.move('cpu');
my_net_fc = copy(my_net);
my_net_conv = copy(my_net);

layer_names = {};
for ii = 1 : 10
    layer_names{ii} = my_net_fc.layers(ii).name;
end
my_net_fc.removeLayer(layer_names);
my_net_fc.rebuild();

layer_names = {};
for ii = 11 : numel(my_net.layers)
    layer_names{ii-10} = my_net_conv.layers(ii).name;
end
my_net_conv.removeLayer(layer_names);
my_net_conv.rebuild();

my_net.move('gpu');

my_net_fc.rebuild();

