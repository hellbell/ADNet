% INIT_PARAMS Set parameters
% 
% Sangdoo Yun, 2017.

% parameter settings
show_visualization = 0;
record_video = 0;
GT_anno_interval = 1;

% ============================
% NETWORK PARAMETERS
% ============================
opts.train_dbs =  {'vot15', 'vot14', 'vot13'};
opts.test_db = 'otb';
opts.train.weightDecay = 0.0005;
opts.train.momentum = 0.9 ;
opts.train.learningRate = 10e-5;
opts.train.conserveMemory = true ;
opts.minibatch_size = 32;
opts.numEpoch = 30;
opts.numInnerEpoch = 3;
opts.continueTrain = false;
opts.samplePerFrame_large = 40;
opts.samplePerFrame_small = 10;
opts.inputSize = [112,112,3];
opts.stopIou = 0.93;
opts.meta.inputSize = [112, 112, 3];
opts.train.gt_skip = 1;
opts.train.rl_num_batches = 5;
opts.train.RL_steps = 10;
opts.use_finetune = true;
opts.scale_factor = 1.05;

% test
opts.finetune_iters = 20;
opts.finetune_iters_online = 10;
opts.finetune_interval = 30;
opts.posThre_init = 0.7;
opts.negThre_init = 0.3;
opts.posThre_online = 0.7;
opts.negThre_online = 0.5;
opts.nPos_init = 200;
opts.nNeg_init = 150;
opts.nPos_online = 30;
opts.nNeg_online = 15;
opts.finetune_scale_factor = 3.0;
opts.redet_scale_factor = 3.0;
opts.finetune_trans = 0.10;
opts.redet_samples = 256;

opts.successThre = 0.5;
opts.failedThre = 0.5;

opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

opts.nPos_train = 150;
opts.nNeg_train = 50;
opts.posThre_train = 0.5;
opts.negThre_train = 0.3;

opts.random_perturb.x = 0.15;
opts.random_perturb.y = 0.15;
opts.random_perturb.w = 0.03;
opts.random_perturb.h = 0.03;
opts.action_move.x = 0.03;
opts.action_move.y = 0.03;
opts.action_move.w = 0.03;
opts.action_move.h = 0.03;

opts.action_move.deltas = [
    -1, 0, 0, 0;  % left
    -2, 0, 0, 0;  % left x2
    +1, 0, 0, 0;  % right
    +2, 0, 0, 0;  % right x2  
    0, -1, 0, 0;  % up
    0, -2, 0, 0;  % up x2 
    0, +1, 0, 0;  % down
    0, +2, 0, 0;  % down x2  
    0,  0,  0,  0 % stop
    0,  0, -1, -1; % smaller
    0,  0, +1, +1; % bigger   
    ];

opts.num_actions = 11;
opts.stop_action = 9;
opts.num_show_actions = 20;
opts.num_action_step_max = 20;
opts.num_action_history = 10;

opts.visualize = true;
opts.printscreen = true;
