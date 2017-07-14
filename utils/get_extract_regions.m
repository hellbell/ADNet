function ims = get_extract_regions(im, boxes, opts)
% GET_EXTRACT_REGIONS extract image regions 
% modified from MDNET_EXTRACT_REGIONS() of MDNet 
% 
% Sangdoo Yun, 2017.

if nargin < 3
    opts.input_size = 112;
    opts.crop_mode = 'wrap';
    opts.crop_padding = 16;
end

num_boxes = size(boxes, 1);

crop_mode = opts.crop_mode;
crop_size = opts.input_size;
crop_padding = opts.crop_padding;

ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');

im_ = single(im);
boxes_ = boxes;
centers = boxes_(:,1:2) + 0.5 * boxes_(:,3:4);
wh = boxes_(:,3:4) * 1.4;
boxes_ = [centers, centers] + [-0.5 * wh, +0.5 * wh];
boxes_(:,1) = max(1, boxes_(:,1));
boxes_(:,2) = max(1, boxes_(:,2));
boxes_(:,3) = min(size(im,2), boxes_(:,3));
boxes_(:,4) = min(size(im,1), boxes_(:,4));

st = 1;
en = min( size(boxes_, 1) , 1000);
while(1)
    box = double(boxes_(st:en, [2,1,4,3]));
    ims(:,:,:,st:en) = gather(cropRectanglesMex( im_, box, [crop_size crop_size]) - 128);
    if en == size(boxes_, 1)
        break;
    end
    st = en + 1;
    en = min(size(boxes_, 1), en + 1000);
end
