clear
clc

N=3;
G=randn(N,2);


% G=[-1 0.1
%    1 0.1
%    2 0.2
%    -2 0.1];
% N=size(G,1);

H=[zeros(2,2) zeros(2,N);zeros(N,2) eye(N,N)];
f=zeros(N+2,1);
A=[-G -eye(N,N);zeros(N,2) eye(N,N)];
b=zeros(2*N,1);
Aeq=[1 zeros(1,2*N-1);zeros(2*N-1,1) zeros(2*N-1,2*N-1)];
beq=[1 zeros(1,2*N-1)];
X = quadprog(H,f,A,b,[],[],-ones(2*N,1),ones(2*N,1));
Xn=X/norm(X(1:2));
figure(1);hold on
plot([zeros(N,1) G(:,1)]',[zeros(N,1) G(:,2)]','r');
plot([0 Xn(1)]',[0 Xn(2)]','b');
plot(0,0,'ko');


% 
% 
% H=[zeros(2,2) zeros(2,4);zeros(4,2) eye(4,4)];
% f=zeros(6,1);
% A=[-G -eye(4,4);zeros(4,2) eye(4,4)];
% b=zeros(8,1);
% X = quadprog(H,f,A,b,[],[],zeros(1,6),ones(1,6));