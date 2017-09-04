function videos = get_benchmark_info(bench_name)
if nargin < 1
    bench_name = 'otb-vot15';
end

bench_path = fullfile('utils/videolist', sprintf('%s.txt', bench_name));
videos = importdata(bench_path);

end