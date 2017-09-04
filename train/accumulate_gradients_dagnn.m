function state = accumulate_gradients_dagnn(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for p=1:numel(net.params)
   
    parDer = net.params(p).der ;
    
    if ~isempty(net.params(p).der)
        switch net.params(p).trainMethod
            
            case 'average' % mainly for batch normalization
                thisLR = net.params(p).learningRate ;
                net.params(p).value = ...
                    (1 - thisLR) * net.params(p).value + ...
                    (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
                
            case 'gradient'
                thisDecay = opts.weightDecay * net.params(p).weightDecay ;
                thisLR = opts.learningRate * net.params(p).learningRate ;

                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.params(p).value) ;
                
                % Update momentum.
                state.momentum{p} = vl_taccum(...
                    opts.momentum, state.momentum{p}, ...
                    -1, parDer) ;
                
                % Nesterov update (aka one step ahead).               
                delta = state.momentum{p} ;
                
                % Update parameters.
                net.params(p).value = vl_taccum(...
                    1,  net.params(p).value, thisLR, delta) ;
                 
            case 'otherwise'
                error('Unknown training method ''%s'' for parameter ''%s''.', ...
                    net.params(p).trainMethod, ...
                    net.params(p).name) ;
        end
    end
end