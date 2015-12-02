% Efficient batch mode semi-supervised active learning algorithm
% Computation complexity: construct matrix A+SSL+QP_solver = O(n^2), so the
% algorithm can handle at most 10,000 data samples. If more data samples
% should be handled, perform K-means first to obtain the landmark data
% Written in 9/28/2011 at UTA
function [L score index] = AL_semi_fast1(A, A2,Yl,label_index,k,r)
% A:  if xi and xj are knn, Aij=1, otherwise, Aij=0
% A2: n*n Gaussian kernel matrix, should be positive semi-definite, 
% A2_{ij} = exp (-|xi-xj|^2/sigmma);
% Yl: l*c label matrix
% label_index: l*1 label index vector
% k: selected data number
% r: algorithm parameter
% score: obtained score of data samples
% index: selected data index
% r = [10^-5,10^-3,..,10^3,10^5]
n = size(A);
A=A+0.000001*ones(n);
%A = sparse(A);
d = sum(A,2);
%D = spdiags(d,0,n,n);
D = diag(d);
L = D - A;
unlabel_index = 1:n;
unlabel_index(label_index) = [];
Lul = L(unlabel_index,label_index);
Luu = L(unlabel_index,unlabel_index);
F(label_index,:) = Yl;
F(unlabel_index,:) = -Luu\(Lul*Yl);
F = F+eps;
%b = (r*sum(F.*log(F),2));
%b(label_index) = 0;
%b = r*sum(F.*F,2);
maxentropy = log(1/size(F,2)); 
b = (r*sum(F.*log(F),2))/maxentropy;
b(label_index) = 0;
tic; [x obj] = QP_solver_fast(A2+0*eye(n),b,k); toc;
[score index] = sort(x,'descend');

plot(obj);

% min_{x>=0,x'*1=k} 0.5*x'*A*x+x'*b
% A should be positive semi-definite
function [x obj] = QP_solver_fast(A,b,k)

NITER = 300;
obj(1:NITER) = 0;
mu = 0.000001;
rho = 1.1;

n = length(b);
lambda1 = 0;
Lambda2 = zeros(n,1);
v = zeros(n,1);
x = 1/n*ones(n,1);
for iter = 1:NITER
    AA = A + mu*eye(n) + mu*ones(n);
    bb = mu*v+(mu*k-lambda1)*ones(n,1)-Lambda2-b;
    x = (AA)\bb;
%     for it=1:5
%     xg = AA*x - bb; s = (xg'*xg)/(xg'*AA*xg); x = x - s*xg;
%     end;
    xx = x+1/mu*Lambda2;
    v = max(xx,0);
    lambda1 = lambda1 + mu*(sum(x)-k);
    Lambda2 = Lambda2 + mu*(x-v);
    mu = mu*rho;
    
    obj(iter) = 0.5*x'*A*x+x'*b;
end;

iiii=1;




