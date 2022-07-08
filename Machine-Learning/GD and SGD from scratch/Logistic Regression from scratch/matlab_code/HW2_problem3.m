filename = 'HW2_sample_data_Pr3.txt';
delimiterIn = ' ';
data = importdata(filename,delimiterIn);

t = array2table(data, 'VariableNames', {'instance_number', 'x1','x2', 'label'}) ;
t.x0(:,1)=1;  %adding bias to the table
t.label_map(:,1)=t.label;
t.label_map(t.label==2)=0;
t = t(:,[1 5 2 3 4 6]);  % rearraging the table
disp(t)

%d features ---> bias+ features (This is what we are optimizing)
%N samples
%y label
%wk weights dX1
%my_sigmoid returns the value of the discrimant function
%g_new is the value (1XN) kind of our new prediction from the model
%mycost is the value of the objective function
%gradint is the drivative of the mycost function (dX1)

x = table2array(t(1:end,2:4))';  % d X N
y = table2array(t(1:end,6))';    % 1 X N
wk = [2.5;
        4;
        6.8];

fprintf('size of x: d X N --> %d X %d\n', size(x));
fprintf('size of w: 1 X d --> %d X %d\n', size(wk));
fprintf('size of y: N X 1 --> %d X %d\n', size(y));



eta =0.05;
J=mycost(wk,x,y); %returns a scaler; Calc
epsilon = 0.01;
gradient=[ones(3,1)] %d X 1  %Initialize the gradient
while max(abs(gradient))> epsilon
 g_new=mysigmoid(x,wk); %returns 1 X N
 disp(size(g_new')) %retursn N X 1
 gradient= (1/20)*x*(g_new'-y'); % (1/N)*(dXN) *(N*1)
 wk=wk-eta*gradient; %(dX1) - learning_rate(dX1)
 disp(wk)
 J=[J mycost(wk,x,y)];
end

%disp(J)
plot(J, 'LineWidth', 1.5)
xlabel("iterations");
ylabel("J (Cost)");
