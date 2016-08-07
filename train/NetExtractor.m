%%this is a tool used to extract the net configure from the matlab model
%and comply to the format I speced.
%written by oar, 30/04/2016

f = fopen('output/test.cnet','w');
fprintf(f,'7\n');
fprintf(f,'28 28 1\n\n');

for i = 1:7
    if lenet.layers{i+1}.type(1) == 'C'
        if lenet.layers{i+1}.type(2) ~= '5'
        fprintf(f,'0\n');
        fprintf(f,'%d\n',lenet.layers{i+1}.outputmaps);
        fprintf(f,'%d\n',lenet.layers{i}.outputmaps);
        fprintf(f,'%d\n%d\n',lenet.layers{i+1}.kernelsize,lenet.layers{i+1}.kernelsize);
        fprintf(f,'1\n');
        for j=1:lenet.layers{i+1}.outputmaps
            fprintf(f,'\t%d\n',lenet.layers{i}.outputmaps);
            for k=1:lenet.layers{i}.outputmaps
                fprintf(f,'\t\t%d ',k-1);
                for m = 5:-1:1
                    for n =5:-1:1
                        fprintf(f,'%f ',lenet.layers{i+1}.w{k}{j}(m,n));
                    end
                end
                fprintf(f,'\n');
            end
            fprintf(f,'\t\t%d\n',lenet.layers{i+1}.b{j});
        end
        fprintf(f,'\n');
        end
    end
    
    if lenet.layers{i+1}.type(1) == 'S'
        if lenet.layers{i+1}.type(2) ~= 'o'
            %lenet.layers{i+1}.type
            fprintf(f,'2\n1\n%d\n%d\n%d\n\n',lenet.layers{i}.outputmaps,lenet.layers{i+1}.scale,lenet.layers{i+1}.scale);
        end
    end
    
    if strcmp(lenet.layers{i+1}.type,'C5')
        fprintf(f,'1\n');
        fprintf(f,'256\n120\n');
        for m=1:256
            for n = 1:120
                fprintf(f,'%f ',lenet.layers{i+1}.w(n,m));
            end
        end
        fprintf(f,'\n');
        for m=1:120
            fprintf(f,'%f ',lenet.layers{i+1}.b(m));
        end
        fprintf(f,'\n\n');
    end
    
    if strcmp(lenet.layers{i+1}.type, 'F6')
        fprintf(f,'1\n120\n84\n');
        for m=1:120
            for n=1:84
                fprintf(f,'%f ',lenet.layers{i+1}.w(n,m));
            end
        end
        fprintf(f,'\n');
        for m=1:84
            fprintf(f,'%f ',lenet.layers{i+1}.b(m));
        end
        fprintf(f,'\n\n');
    end
    
    if strcmp(lenet.layers{i+1}.type, 'Soft')
        fprintf(f,'1\n84\n10\n');
        for m=1:84
            for n=1:10
                fprintf(f,'%f ',lenet.layers{i+1}.w(n,m));
            end
        end
        fprintf(f,'\n');
        for m=1:10
            fprintf(f,'%f ',lenet.layers{i+1}.b(m));
        end
        fprintf(f,'\n\n');
    end                
end
fclose(f);
