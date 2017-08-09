function build_cropRectanglesMex_on_windows( cudaRoot )
%build_cropRectanglesMex builds package cropRectanglesMex
%
% INPUT:
%   cudaRoot - path to the CUDA installation

% Anton Osokin, firstname.lastname@gmail.com, May 2015

if ~exist('cudaRoot', 'var')
    cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5';
    nvccPath = fullfile(cudaRoot, 'bin', 'nvcc.exe');
    cudaLibPath = fullfile(cudaRoot, 'lib', 'x64');
end

if ~exist(nvccPath, 'file')
    error('NVCC compiler was not found!');
end

root = fileparts( mfilename('fullpath') );

% compiling
compileCmd = [ '"', nvccPath, '"', ...
        ' -c ', '"', fullfile(root,'cropRectanglesMex.cu'), '"', ... 
        ' -I"', fullfile( matlabroot, 'extern', 'include'), '"', ...
        ' -I"', fullfile( matlabroot, 'toolbox', 'distcomp', 'gpu', 'extern', 'include'), '"', ...
        ' -I"', fullfile( cudaRoot, 'include'), '"', ...        
        ' -DNDEBUG -DENABLE_GPU', ...
        ' -Xcompiler /MD', ...
        ' -o "', fullfile(root,'cropRectanglesMex.o'), '"'];
system( compileCmd );

% linking
mopts = {'-outdir', root, ...
         '-output', 'cropRectanglesMex', ...
         ['-L', cudaLibPath], ...
         '-lcudart', '-lnppi', '-lnppc', '-lgpu', ...
         '-largeArrayDims', ...
         fullfile(root,'cropRectanglesMex.o') };
mex(mopts{:}) ;

delete( fullfile(root,'cropRectanglesMex.o') );
        