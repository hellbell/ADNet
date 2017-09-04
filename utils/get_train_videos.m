function train_db = get_train_videos(opts)
% GET_TRAIN_VIDEOS 
% 
% Sangdoo Yun, 2017.

train_db_names = opts.train_dbs;
test_db_name = opts.test_db;

video_names = {};
video_paths = {};
bench_names = {};
for dbidx = 1 : numel(train_db_names)
    bench_name = train_db_names{dbidx};
    path_ = get_benchmark_path(train_db_names{dbidx});
    video_names_ = get_benchmark_info(sprintf('%s-%s',train_db_names{dbidx}, test_db_name));
    video_paths_ = repmat({path_}, [1, numel(video_names_)]);
    video_names(end+1:end+numel(video_names_)) = video_names_;
    video_paths(end+1:end+numel(video_paths_)) = video_paths_;
    bench_names(end+1:end+numel(video_paths_)) = repmat({bench_name}, [1, numel(video_names_)]);
end

train_db.video_names = video_names;
train_db.video_paths = video_paths;
train_db.bench_names = bench_names;
end
