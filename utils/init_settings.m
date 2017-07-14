% INIT_SETTINGS Set paths
% 
% Sangdoo Yun, 2017.

if ispc
    matconvnet_path = 'matconvnet\matlab\vl_setupnn';
    vgg_m_path = 'models\imagenet-vgg-m-conv1-3.mat';
    result_path = 'results';
    model_save_path = 'models';
else
    matconvnet_path = './matconvnet/matlab/vl_setupnn.m';
    vgg_m_path = './models/imagenet-vgg-m-conv1-3.mat';
    result_path = './results';
    model_save_path  = './models';
end

init_params;



