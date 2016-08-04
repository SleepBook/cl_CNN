function net = initialNet(net, x)
    for l = 1:8
        if strcmp(net.layers{l}.type, 'C1')
            mapsize = 24;
            fan_out = 150;
            for j = 1:6
                fan_in = 25;
                net.layers{l}.w{1}{j} = (rand(5)-0.5) * 2;
                net.layers{l}.b{j} = 0;
            end
            inputmaps = 6;
        end

        
        if strcmp(net.layers{l}.type, 'S2')
            for j = 1:6
                net.layers{l}.b{j} = 0;
                net.layers{l}.w{j} = (rand(6,1)-0.5) * 2;
            end
            mapsize = 12;
        end
        
        if strcmp(net.layers{l}.type, 'C3')
            mapsize = 8;
            fan_out = 400;
            fan_in = 150;
            for j = 1:16
                for i = 1:6
                    net.layers{l}.w{i}{j} = (rand(5)-0.5) * 2 * sqrt(6/(fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = 16;
        end
            
        if strcmp(net.layers{l}.type, 'S4')
            for j = 1:16
                net.layers{l}.b{j} = 0;
                net.layers{l}.w{j} = (rand(6,1)-0.5) * 2 * sqrt(6/(16 * 2));
            end
            mapsize = 4;
        end
    
        if strcmp (net.layers{l}.type,'C5')
            fan_out = 120;
            fan_in = 256;
            net.layers{l}.w = (rand(120,256) - 0.5) * 2 * sqrt(6/(fan_in + fan_out));
            net.layers{l}.b = zeros(120,1);
            inputmaps = 120;
        end
        
        if strcmp(net.layers{l}.type, 'F6')
            fan_in = 120;
            fan_out = 84;
            net.layers{l}.w = (rand(84,120)-0.5) * 2 * sqrt(6/(fan_in + fan_out));
            net.layers{l}.b = zeros(84,1);
            inputmaps = 84;
        end
        
        if strcmp(net.layers{l}.type, 'Soft')
            fan_in = 84;
            fan_out = 10;
            net.layers{l}.w = (rand(10,84)-0.5) * 2 * sqrt(6/(fan_in+fan_out));
            net.layers{l}.b = zeros(10,1);
        end
    end
end