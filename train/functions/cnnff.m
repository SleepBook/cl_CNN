function net = cnnff(net, x)
    num = size(x, 3);
    net.layers{1}.a{1} = x;
    % 8 layars
    for l = 2:8
        if strcmp(net.layers{l}.type, 'C1')
            for j = 1:6
                %[28,28,?]-[4,4,0]
                z = zeros(size(x) - [4 4 0]);
                z = z + convn(x, net.layers{l}.w{1}{j}, 'valid');
                %b = net.layers{l}.b{j};
                net.layers{2}.z{j} = z + net.layers{2}.b{j} .* ones(size(z));
                %z1 = reshape(net.layers{l}.z{j},[],40);
                net.layers{2}.a{j} = sigm(net.layers{2}.z{j});
            end
            inputmaps = 6;
        end
        
        if strcmp(net.layers{l}.type, 'C3')
            for j = 1:16
                z = zeros(size(net.layers{3}.a{1}) - [4 4 0]);
                for i = 1:6
                    z = z + convn(net.layers{3}.a{i}, net.layers{l}.w{i}{j}, 'valid');
                end
                %b = net.layers{l}.b{j};
                net.layers{4}.z{j} = z + net.layers{4}.b{j} .* ones(size(z));
                %z1 = reshape(net.layers{l}.z{j},[],40);
                net.layers{4}.a{j} = sigm(net.layers{4}.z{j});
            end
            inputmaps = 16;
        end
        
        if strcmp(net.layers{l}.type , 'S2')
            for j = 1:6
                z = convn(net.layers{2}.a{j}, ones(2) / 4,'valid');
                net.layers{3}.a{j} = z(1:2:end, 1:2:end,:);
                %z2 = reshape(net.layers{l}.a{j},[],40);
            end
        end
        
        if strcmp(net.layers{l}.type, 'S4');
            for j = 1:16
                z = convn(net.layers{4}.a{j}, ones(2) / 4,'valid');
                net.layers{5}.a{j} = z(1:2:end, 1:2:end,:);
                %z2 = reshape(net.layers{l}.a{j},[],40);
            end
        end
        
        if strcmp(net.layers{l}.type, 'C5')
            net.layers{5}.fv = [];
            for j = 1:16
                sa = size(net.layers{5}.a{j});
                net.layers{5}.fv = [net.layers{5}.fv; reshape(net.layers{5}.a{j}, 16, sa(3))];
            end
            net.layers{l}.z = net.layers{6}.w * net.layers{5}.fv + repmat(net.layers{l}.b,1,num);
            net.layers{l}.a = sigm(net.layers{l}.z);
            inputmaps = 120;
        end
        
        if strcmp (net.layers{l}.type, 'F6')
            net.layers{l}.z = net.layers{7}.w * net.layers{6}.a + repmat(net.layers{7}.b,1,num);
            net.layers{l}.a = sigm(net.layers{l}.z);
        end
        
        if strcmp(net.layers{l}.type,'Soft')
            net.layers{8}.z = net.layers{8}.w * net.layers{7}.a + repmat(net.layers{8}.b,1,num);
%             net.layers{8}.z = bsxfun(@minus, net.layers{l}.z, max(net.layers{l}.z,[],1));
%             net.layers{8}.z = exp(net.layers{8}.z);
%             net.layers{l}.a = bsxfun(@rdivide, net.layers{l}.z, sum(net.layers{l}.z));
            net.layers{8}.a = sigm(net.layers{8}.z);
        end
    end
    
end