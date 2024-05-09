% common codes
m=512;n=512;
image=imread('./test_images/512_512_circles.png');
gimage=rgb2gray(image);
u=reshape(gimage,m*n,1);
u=double(u)/255;
mask=imread('./test_masks/512_512_random70.png');
mask=reshape(mask,m*n,1);
I=find(mask~=0);
s=length(I);
A=sparse(1:s,[I],ones(s,1),s,m*n);
b=A*u; % color information

% D
k = 2*m*n - m - n;

Positive1 = [[1:m*n],[1:m*(n-1)]];
Negative1 = [[1:m*n],[n+1:m*n]];

Positive1(m:n:m*n) = [];
Negative1(1:n:m*n) = [];

D = sparse([1:k].',Positive1,1,k,m*n)+sparse([1:k].',Negative1,-1,k,m*n);

%%% Reformulate

%interior-point
tic

x = optimvar('x',m*n,1);
y = optimvar('y',k,1);
prob = optimproblem('Objective',sum(y),'ObjectiveSense','min');
prob.Constraints.c1 = A*x-b==0;
prob.Constraints.c2 = D*x-y<=0;
prob.Constraints.c3 = -D*x-y<=0;
problem = prob2struct(prob);
problem.options=optimoptions("linprog",'Algorithm','interior-point',"Display","iter","OptimalityTolerance",1e-8,"ConstraintTolerance",1e-4);
x = linprog(problem);

toc
PSNR = 10*log(m*n/norm(x(1:m*n,1)-u,2)^2)/log(10)

sol = x(1:m*n,1);
abs(sum(D*sol))


 % show image
xx = reshape(x(1:m*n,1),m,n)*255;
imshow(uint8(xx))

% dual

tic

[s,o] = size(b);
M = speye(k);
AA = sparse([D',-D',A';-M,-M,sparse(k,s)]);
AAA = sparse([1:2*k],[1:2*k],ones(1,2*k),2*k,2*k+s);
options = optimoptions('linprog','Algorithm','interior-point');
f = (-1)*sparse(1,[(2*k+1):(2*k+s)],b',1,2*k+s);
[z,fval,exitflag,output] = linprog(f,AAA,zeros(2*k,1),AA,sparse([m*n+1:m*n+k],1,ones(1,k),m*n+k,1),[],[],options)

toc




%%% dual-simplex
tic

x = optimvar('x',m*n,1);
y = optimvar('y',k,1);
prob = optimproblem('Objective',sum(y),'ObjectiveSense','min');
prob.Constraints.c1 = A*x-b==0;
prob.Constraints.c2 = D*x-y<=0;
prob.Constraints.c3 = -D*x-y<=0;
problem = prob2struct(prob);
problem.options=optimoptions("linprog",'Algorithm','dual-simplex',"Display","iter","OptimalityTolerance",1e-8,"ConstraintTolerance",1e-4);
x = linprog(problem);

toc
PSNR = 10*log(m*n/norm(x(1:m*n,1)-u,2)^2)/log(10)

sol = x(1:m*n,1);
abs(sum(D*sol))

% dual

tic

[s,o] = size(b);
numbers = 1:k;
M = sparse(numbers,numbers,1,k,k);
AA = sparse([D',-D',A';-M,-M,sparse(k,s)]);
AAA = sparse([1:2*k],[1:2*k],ones(1,2*k),2*k,2*k+s);
options = optimoptions('linprog','Algorithm','dual-simplex');
f = (-1)*sparse(1,[(2*k+1):(2*k+s)],b',1,2*k+s);
[z,fval,exitflag,output] = linprog(f,AAA,zeros(2*k,1),AA,sparse([m*n+1:m*n+k],1,ones(1,k),m*n+k,1),[],[],options)

toc


 % show image
xx = reshape(x(1:m*n,1),m,n)*255;
imshow(uint8(xx))