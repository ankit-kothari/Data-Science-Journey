function g_new = mysigmoid(x,w)
wT_x  = 1*w'*x;
g_new = 1./(1+exp(-1*wT_x));
end