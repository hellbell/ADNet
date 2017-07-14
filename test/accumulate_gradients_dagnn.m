% -------------------------------------------------------------------------
function state = accumulate_gradients_dagnn(net, state, params, batchSize)
% -------------------------------------------------------------------------
% numGpus = numel(params.gpus) ;
% otherGpus = setdiff(1:numGpus, labindex) ;
 
for p=1:numel(net.params)
 
  parDer = net.params(p).der ;
  if isempty(parDer)
%       parDer = gpuArray([]);
      continue;
  end
 
  switch net.params(p).trainMethod
 
    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;
 
    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;
 
      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;
 
        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;
 
          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end
 
          % Update parameters.
          net.params(p).value = vl_taccum(...
            1,  net.params(p).value, thisLR, delta) ;
 
        else
          % call solver function to update weights
          [net.params(p).value, state.solverState{p}] = ...
            params.solver(net.params(p).value, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end


% function state = accumulate_gradients_dagnn(state, net, opts, batchSize, mmap)
% % -------------------------------------------------------------------------
% for p=1:numel(net.params)
%     
% %     if ~strcmp(net.params(p).name(end), 'Z')
% %         continue;
% %     end
%     %   % bring in gradients from other GPUs if any
%     %   if ~isempty(mmap)
%     %     numGpus = numel(mmap.Data) ;
%     %     tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
%     %     for g = setdiff(1:numGpus, labindex)
%     %       tmp = tmp + mmap.Data(g).(net.params(p).name) ;
%     %     end
%     %     net.params(p).der = net.params(p).der + tmp ;
%     %   else
%     %     numGpus = 1 ;
%     %   end
%     
%     parDer = net.params(p).der ;
%     
%     if ~isempty(net.params(p).der)
%         switch net.params(p).trainMethod
%             
%             case 'average' % mainly for batch normalization
%                 thisLR = net.params(p).learningRate ;
%                 net.params(p).value = ...
%                     (1 - thisLR) * net.params(p).value + ...
%                     (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
%                 
%             case 'gradient'
%                 thisDecay = opts.weightDecay * net.params(p).weightDecay ;
%                 thisLR = opts.learningRate * net.params(p).learningRate ;
% 
%                 % Normalize gradient and incorporate weight decay.
%                 parDer = vl_taccum(1/batchSize, parDer, ...
%                     thisDecay, net.params(p).value) ;
%                 
%                 % Update momentum.
%                 state.momentum{p} = vl_taccum(...
%                     opts.momentum, state.momentum{p}, ...
%                     -1, parDer) ;
%                 
%                 % Nesterov update (aka one step ahead).               
%                 delta = state.momentum{p} ;
%                 
%                 % Update parameters.
%                 net.params(p).value = vl_taccum(...
%                     1,  net.params(p).value, thisLR, delta) ;
%                 
%                 
% %                 state.momentum{p} = opts.momentum * state.momentum{p} ...
% %                     - thisDecay * net.params(p).value ...
% %                     - (1 / batchSize) * net.params(p).der ;
% %                 net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;
%                 
%             case 'otherwise'
%                 error('Unknown training method ''%s'' for parameter ''%s''.', ...
%                     net.params(p).trainMethod, ...
%                     net.params(p).name) ;
%         end
%     end
% end