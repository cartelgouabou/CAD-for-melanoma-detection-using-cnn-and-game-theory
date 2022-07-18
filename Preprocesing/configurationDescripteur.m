% Multi-Level Histogram of Color Multi-Scale Local Binary Pattern with Spatial Pyramid.

%        spyr                    Spatial Pyramid (nspyr x 4) (default [1 , 1 , 1 , 1] with nspyr = 1)
% 

%        scale                   Multi-Scale vector (1 x nscale) (default scale = 1) where scale(i) = s is the size's factor to apply to each 9 blocks
%                                in the LBP computation, i = 1,...,nscale

%        color                   0 : force gray-scale (dimcolor = 1, default), 1 : RGB-LBP (dimcolor = 3), 2 : nRGB-LBP (dimcolor = 3), 3 : Opponent-LBP (dimcolor = 3), 
%                                4 : nOpponent-LBP (dimcolor = 2), 5 : Hue-LBP (dimcolor = 1)

% 	     maptable                Mapping table for LBP codes. maptable = 0 <=> normal LBP = {0,...,255} (default), 
% 	                             maptable = 1 <=> uniform LBP = {0,...,58}, maptable = 2 <=> uniform rotation-invariant LBP = = {0,...,9}

%        improvedLBP             0 for default 8 bits encoding, 1 for 9 bits encoding (8 + central area)

%        rmextremebins           Force to zero bin = {0 , {255,58,9}} if  rmextremebins = 1 (default rmextremebins = 1)
% 

%        norm                    Normalization : norm = 0 <=> no normalization, norm = 1 <=> v=v/(sum(v)+epsi), norm = 2 <=> v=v/sqrt(sum(v²)+epsi²), 
%                                norm = 3 <=> v=sqrt(v/(sum(v)+epsi)) , norm = 3 <=> L2-clamped (default norm = 1)

%        clamp                   Clamping value (default clamp = 0.2)


function options = configurationDescripteur()
options.spyr          = [1 , 1 , 1 , 1 ; 1/2 , 1/2 , 1/4 , 1/4];
options.scale         = [1 , 2 , 3 , 4 ];
options.rmextremebins = 1;
options.norm          = [0 , 0 , 0];
options.clamp         = 0.2; 
options.maptable = 1;
options.dimcolor = 3;
options.nbbins = 256;
options.size = 30720;
options.improveLBP = 0;
options.nH = 10;
options.color = 1;

nS = length(options.scale)*sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));
% options.size  = length(options.scale)*sum(floor(((1 - options.spyr(:,1))./(options.spyr(:,3)) + 1)).*floor((1 - options.spyr(:,2))./(options.spyr(:,4)) + 1));

 options = {options};
end