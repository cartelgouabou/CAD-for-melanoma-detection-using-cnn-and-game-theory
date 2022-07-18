clear all
close all

img_folders = dir('D:\PROJET\DERMA_ARTICLE\base\mednote\MAL\N\*.jpg');
 
 %R=I(:,:,2);
 %G=I(:,:,2);
 N= length( img_folders );
 for i= 1: N
    img_folders(i).name;
    filename=['D:\PROJET\DERMA_ARTICLE\base\mednote\MAL\N\',img_folders(i).name];
    I=imread(filename);
    imgTreat = colorConstancy(I, 'gray world seg',2);
    FileName = strcat(img_folders(i).name(1:end-4),'.jpg');    
    imwrite(imgTreat,['D:\PROJET\DERMA_ARTICLE\base\mednote\MAL\A\',FileName]); 
end ;