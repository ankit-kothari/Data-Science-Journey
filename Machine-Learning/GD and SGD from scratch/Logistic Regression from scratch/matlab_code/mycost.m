function [J]=mycost(w,x,y)
 g_new = mysigmoid(x,w).';
 temp =y*log(g_new);
 temp=temp+ (1-y)*log(1-g_new);
 [d,N]=size(x);
 J=(-1/N)*temp;



A = [.5 .33 0.1667 0 0 0 0;
    .5 .5 0 0 0 0 0 ;
    .2 .6 .2 0 0 0 0;
    0 0 .5 .33 0.1667 0 0;
    0 0 .2 0 .6 .2 0;
    0 0 0 0 0  .5 .5;
    0 0 0 0 0 .25 .74];
