function [ feat, ims ] = get_conv_feature(net, img, boxes, opts)
% GET_CONV_FEATURE extract convolutional feature of image patches
% 
% Sangdoo Yun, 2017.

opts.input_size = 112;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;
opts.batchSize_test = 128;

n = size(boxes,1);
ims = get_extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);

for i=1:nBatches
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    
    batch = gpuArray(batch);
    
    inputs = {'input', batch};    
    net.eval(inputs);
    f = net.vars(end).value;
    f = gather(f);
    
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
end