clc;
clear all;
close all;
pathDest = 'D:\PROJET\Class_imbalance\DATASET';


dataDir1 =  fullfile('D:','PROJET','BASE','BASE','ISIC2018','ACK');
data1 = datastore(dataDir1);
path1=char(data1.Files(1));
img1=imread(path1);
pos1=34
path1(pos1:end)

%Test
for i = 1:7
    if i==1
        temp='ACK';
    elseif i==2
        temp='BCC';
    elseif i==3
        temp='BEK';
    elseif i==4
        temp='DEF';
    elseif i==5
        temp='MEL';
    elseif i==6
        temp='NEV';
    else
        temp='VAL';
    end
    dataDir =  fullfile('D:','PROJET','BASE','BASE','ISIC2018',temp);
    data = datastore(dataDir);
    num = numel(data.Files); 
    for j = 1: num
        path=char(data.Files(j));
        img=imread(path);
        filename=path(pos1:end);
        imgTreat = colorConstancy(img, 'gray world seg',2);
        fullFileName = fullfile(pathDest,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
end
