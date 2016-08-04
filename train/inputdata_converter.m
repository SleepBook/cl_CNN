%this is a tools used to generate the batch test data.

data_op = 0;
batch_mode = 1;
batch_size = 20;

if(data_op == 1)
    f = fopen('output/test.cdat','w');
    fprintf(f,'1\n28\n28\n');
    for m=1:28
        for n=1:28
            fprintf(f,'%f ',testImages(m,n,1));
        end
    end
    fprintf(f, '\n');
    fclose(f);
end

if(batch_mode == 1)
    id = fopen('output/batch_test.cdat','w');
    fprintf(id,'1\n28\n28\n');
    for i=1:batch_size
    for m=1:28
        for n=1:28
            fprintf(id,'%f ',testImages(m,n,i));
        end
    end
    fprintf(id,'\n');
    end
    fclose(id);

    lb = fopen('output/batch_test.lbl','w');
    for i=1:batch_size
        if testLabels(i)==10
            fprintf(lb,'0\n');
        else   
            fprintf(lb,'%d\n',testLabels(i));
        end
    end
    fclose(lb);
end