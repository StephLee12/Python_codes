clear all;
clc;

%input vector P
P = [-0.3 -0.8 -0.5 0.1 0.9 0.4;
    0.5 -0.1 0.3 0.1 -0.5 0.3];
% target vector T
T = [1 1 1 0 0 0];
plotpv(P,T);
pause;

% initialize a network
net = newp([-1 1;-1 1],1);
net = init(net);
y = sim(net,P); %simulation
e = T - y;
w = net.iw{1,1}; % get weight
b = net.b{1}; % get bias
plotpc(w,b);
pause;     

while (mae(e) > 0.001)
    dw = learnp(w,P,[],[],[],[],e,[],[],[],[],[]);
    db = learnp(b,ones(1,6),[],[],[],[],e,[],[],[],[],[]);
    w = w + dw;
    b = b + db;
    net.iw{1,1} = w;
    net.b{1} = b;
    plotpc(w,b);
    pause;
    y = sim(net,P);
    e = T - y;
end

