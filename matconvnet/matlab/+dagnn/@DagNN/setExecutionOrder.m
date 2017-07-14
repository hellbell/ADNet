function ord = setExecutionOrder(obj, order)
% purpose: to modify execution order 
% 

obj.executionOrder = order;
ord = obj.getLayerExecutionOrder;