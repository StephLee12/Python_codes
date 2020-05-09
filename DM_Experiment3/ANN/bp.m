clear all;
clc;
% input vector
P = [0 1 2 3 4 5 6 7 8 9 10];
% target vector
T = [0 1 2 3 4 3 2 1 2 3 4];
%initialize a feed-forward network
net = newff([0 10],[5 1],{'tansig','purelin'},'trainlm');
y1 = sim(net,P);
subplot(1,2,1);
plot(P,T,'o',P,y1,'x');
% set training params
net.trainParam.epochs = 100;
net.trainParam.goal = 0.005;
% train
net = train(net,P,T);
y2 = sim(net,P);
subplot(1,2,2)
plot(P,T,'o',P,y1,'x',P,y2,'*');
testData = 6;
y3  = sim(net,testData)