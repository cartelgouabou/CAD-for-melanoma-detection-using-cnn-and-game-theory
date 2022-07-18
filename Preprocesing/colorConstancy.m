%Inspired from article Improving dermoscopy image clarassification using
%color constancy of C Barata and al.
% E[Er,Eg,Eb] is the illuminant vector
function [OUT] = colorConstancy(I, alargorithm,s,p) %, norm

    [hau,lar,~]=size(I);
    switch (alargorithm)
        case 'gray world'
            kEr      = sum(sum(I(:,:,1)))/(hau*lar);                 
            kEg      = sum(sum(I(:,:,2)))/(hau*lar);
            kEb      = sum(sum(I(:,:,3)))/(hau*lar);
            k        = sqrt(kEr^2 + kEg^2  + kEb^2); %k is the normalarization constant that ensures that e has unit length with respect to the euclidean norm
            OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
            OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
            OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
            OUT = uint8(OUT);
            
        case 'shades of gray'
            
            if p == 4
             kEr      = nthroot(sum(sum((I(:,:,1).^4)))/(hau*lar),4);
             kEg      = nthroot(sum(sum((I(:,:,2).^4)))/(hau*lar),4);
             kEb      = nthroot(sum(sum((I(:,:,3).^4)))/(hau*lar),4);
             k        = sqrt(kEr^2 + kEg^2  + kEb^2); %k is the normalarization constant that ensures that e has unit length with respect to the euclidean norm
             OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
             OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
             OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
             OUT = uint8(OUT);
            elseif p == 6
             kEr      = nthroot(sum(sum((I(:,:,1).^6)))/(hau*lar),6);
             kEg      = nthroot(sum(sum((I(:,:,2).^6)))/(hau*lar),6);
             kEb      = nthroot(sum(sum((I(:,:,3).^6)))/(hau*lar),6);
             k        = sqrt(kEr^2 + kEg^2  + kEb^2); %k is the normalarization constant that ensures that e has unit length with respect to the euclidean norm
             OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
             OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
             OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
             OUT = uint8(OUT);
            else
               OUT = I; disp ('Error, shades of gray must have a parameter p, p == 4 or 6.');
            end

       case 'shades of gray seg'      %segmenting the image and computing the average color of all segments
           
           if s == 1
               if p == 4
                meanSeg = (hau/2)*(lar/2);
                mean = 2^2;
                kEr  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),1).^4))/(meanSeg),4) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),1).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),1).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),1).^4))/(meanSeg),4))/mean;
                kEg  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),2).^4))/(meanSeg),4) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),2).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),2).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),2).^4))/(meanSeg),4))/mean;
                kEb  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),3).^4))/(meanSeg),4) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),3).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),3).^4))/(meanSeg),4) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),3).^4))/(meanSeg),4))/mean;
                k    = sqrt(kEr^2 + kEg^2  + kEb^2); 
                OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
                OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
                OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
                OUT = uint8(OUT);
                
               elseif p == 6
                meanSeg = (hau/2)*(lar/2);
                mean = 2^2;
                kEr  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),1).^6))/(meanSeg),6) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),1).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),1).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),1).^6))/(meanSeg),6))/mean;
                kEg  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),2).^6))/(meanSeg),6) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),2).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),2).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),2).^6))/(meanSeg),6))/mean;
                kEb  = (nthroot(sum(sum(I(1: round(hau/2),1:round(lar/2),3).^6))/(meanSeg),6) + nthroot(sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),3).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),3).^6))/(meanSeg),6) + nthroot(sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),3).^6))/(meanSeg),6))/mean;
                k    = sqrt(kEr^2 + kEg^2  + kEb^2); 
                OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
                OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
                OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
                OUT = uint8(OUT); 
                
               else
                OUT = I; disp ('Error, gray world seg must have a parameter p,p=4 or p=6.');
               end
           else
               OUT = I; disp ('Error, gray world seg must have 2 parameters s, s=1 and p=4 or 6.');
           end
           
        case 'gray world seg'      %segmenting the image and computing the average color of all segments
           
           if s == 1 
            meanSeg = (hau/2)*(lar/2);
            mean = 2^2;
            kEr      = (sum(sum(I(1:round(hau/2),1:round(lar/2),1)))/(meanSeg) + sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),1)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),1)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),1)))/(meanSeg))/mean;
            kEg      = (sum(sum(I(1:round(hau/2),1:round(lar/2),2)))/(meanSeg) + sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),2)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),2)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),2)))/(meanSeg))/mean;
            kEb      = (sum(sum(I(1:round(hau/2),1:round(lar/2),3)))/(meanSeg) + sum(sum(I(1:round(hau/2),round(lar/2+1):(lar),3)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),1:round(lar/2),3)))/(meanSeg) + sum(sum(I(round(hau/2+1):(hau),round(lar/2+1):(lar),3)))/(meanSeg))/mean;
            k        = sqrt(kEr^2 + kEg^2  + kEb^2); 
            OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
            OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
            OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
            OUT = uint8(OUT);
           elseif s == 2 
            meanSeg = (hau/4)*(lar/4);
            mean = 2^4;
            kEr      = (sum(sum(I(1: round(hau/4),1:round(lar/4),1)))/(meanSeg) + sum(sum(I(1:round(hau/4),round(lar/4+1):round(lar/2),1)))/(meanSeg)+ sum(sum(I(1:round(hau/4),round(lar/2+1):round((3*lar/4)),1)))/(meanSeg) + sum(sum(I(1:round(hau/4),round((3*lar/4)+1):(lar),1)))/(meanSeg) ...
                       +sum(sum(I(round(hau/4+1): round(hau/2),1:round(lar/4),1)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round(lar/4+1):round(lar/2),1)))/(meanSeg)+ sum(sum(I(round(hau/4+1): round(hau/2),round(lar/2+1):round((3*lar/4)),1)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round((3*lar/4)+1):(lar),1)))/(meanSeg)... 
                       +sum(sum(I(round(hau/2+1): round(3*hau/4),1:round(lar/4),1)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/4+1):round(lar/2),1)))/(meanSeg)+ sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/2+1):round((3*lar/4)),1)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round((3*lar/4)+1):(lar),1)))/(meanSeg)...
                       +sum(sum(I(round((3*hau/4)+1): round(hau),1:round(lar/4),1)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/4+1):round(lar/2),1)))/(meanSeg)+ sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/2+1):round((3*lar/4)),1)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round((3*lar/4)+1):(lar),1)))/(meanSeg))/mean;
            kEg      = (sum(sum(I(1: round(hau/4),1:round(lar/4),2)))/(meanSeg) + sum(sum(I(1:round(hau/4),round(lar/4+1):round(lar/2),2)))/(meanSeg)+ sum(sum(I(1:round(hau/4),round(lar/2+1):round((3*lar/4)),2)))/(meanSeg) + sum(sum(I(1:round(hau/4),round((3*lar/4)+1):(lar),2)))/(meanSeg)...
                      + sum(sum(I(round(hau/4+1): round(hau/2),1:round(lar/4),2)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round(lar/4+1):round(lar/2),2)))/(meanSeg)+ sum(sum(I(round(hau/4+1): round(hau/2),round(lar/2+1):round((3*lar/4)),2)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round((3*lar/4)+1):(lar),2)))/(meanSeg)...
                      + sum(sum(I(round(hau/2+1): round(3*hau/4),1:round(lar/4),2)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/4+1):round(lar/2),2)))/(meanSeg)+ sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/2+1):round((3*lar/4)),2)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round((3*lar/4)+1):(lar),2)))/(meanSeg)...
                      + sum(sum(I(round((3*hau/4)+1): round(hau),1:round(lar/4),2)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/4+1):round(lar/2),2)))/(meanSeg)+ sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/2+1):round((3*lar/4)),2)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round((3*lar/4)+1):(lar),2)))/(meanSeg))/mean;
          
            kEb      = (sum(sum(I(1: round(hau/4),1:round(lar/4),3)))/(meanSeg) + sum(sum(I(1:round(hau/4),round(lar/4+1):round(lar/2),3)))/(meanSeg)+ sum(sum(I(1:round(hau/4),round(lar/2+1):round((3*lar/4)),3)))/(meanSeg) + sum(sum(I(1:round(hau/4),round((3*lar/4)+1):(lar),3)))/(meanSeg)...
                      + sum(sum(I(round(hau/4+1): round(hau/2),1:round(lar/4),3)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round(lar/4+1):round(lar/2),3)))/(meanSeg)+ sum(sum(I(round(hau/4+1): round(hau/2),round(lar/2+1):round((3*lar/4)),3)))/(meanSeg) + sum(sum(I(round(hau/4+1): round(hau/2),round((3*lar/4)+1):(lar),3)))/(meanSeg)...
                      + sum(sum(I(round(hau/2+1): round(3*hau/4),1:round(lar/4),3)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/4+1):round(lar/2),3)))/(meanSeg)+ sum(sum(I(round(hau/2+1): round(3*hau/4),round(lar/2+1):round((3*lar/4)),3)))/(meanSeg) + sum(sum(I(round(hau/2+1): round(3*hau/4),round((3*lar/4)+1):(lar),3)))/(meanSeg)...
                      + sum(sum(I(round((3*hau/4)+1): round(hau),1:round(lar/4),3)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/4+1):round(lar/2),3)))/(meanSeg)+ sum(sum(I(round((3*hau/4)+1): round(hau),round(lar/2+1):round((3*lar/4)),3)))/(meanSeg) + sum(sum(I(round((3*hau/4)+1): round(hau),round((3*lar/4)+1):(lar),3)))/(meanSeg))/mean;
              k        = sqrt(kEr^2 + kEg^2  + kEb^2); 
            OUT(:,:,1) = (k/(kEr*sqrt(3)))*double(I(:,:,1));
            OUT(:,:,2) = (k/(kEg*sqrt(3)))*double(I(:,:,2));
            OUT(:,:,3) = (k/(kEb*sqrt(3)))*double(I(:,:,3));
            OUT = uint8(OUT);
           else
               OUT = I; disp ('Error, gray world seg must have a parameter s, 0<s<3.');
           end
           
       otherwise
           OUT = I; disp ('Error, Unknown alargorithm, please check name.');
    end
end



