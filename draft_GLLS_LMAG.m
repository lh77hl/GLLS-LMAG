function [X, E, obj, iter] = draft_GLLS_LMAG(M, opts)
% Solve the GLLS-LMAG problem 
% ---------------------------------------------
% Input:
%          M    -    n1*n2*n3 tensor
%       opts    -    Necessary parameters

% Output:
%       X       -    n1*n2*n3 tensor
%       E       -    n1*n2*n3 tensor
%       obj     -    objective function value
%       iter    -    number of iterations

%%  
dim = size(M);
d   = ndims(M);

transform  = 'DFT';
for i = 3:d
transform_matrices{i-2} = dftmtx(dim(i)); 
end

lambda     = 0.02;
directions = 1:d; 
tol        = 1e-4; 
max_iter   = 200;
rho        = 1.1;
mu         = 1e-4;
mu1        = 1e-5;
max_mu     = 1e12;
detail     = 1;

if ~exist('opts', 'var')
    opts = [];
end   
if isfield(opts, 'transform');          transform          = opts.transform;          end
if isfield(opts, 'transform_matrices'); transform_matrices = opts.transform_matrices; end
if isfield(opts, 'directions');         directions         = opts.directions;         end
if isfield(opts, 'tol');                tol                = opts.tol;                end
if isfield(opts, 'max_iter');           max_iter           = opts.max_iter;           end
if isfield(opts, 'rho');                rho                = opts.rho;                end
if isfield(opts, 'mu');                 mu                 = opts.mu;                 end
if isfield(opts, 'max_mu');             max_mu             = opts.max_mu;             end
if isfield(opts, 'detail');             detail             = opts.detail;             end

%% initialization
n = length(directions);
X        = randn(dim);
E        = zeros(dim);
L1       = zeros(dim);
L2       = zeros(dim);
L3       = zeros(dim);
Y2_1     = zeros(dim);
Y2_2     = zeros(dim);
Y2_3     = zeros(dim);
b        = zeros(dim);
Lambda   = zeros(dim);
weightTenT = ones(dim);
change=zeros(1,max_iter);
for i = 1:n
    index        = directions(i);
    G{index}     = porder_diff(X,index); 
    Gamma{index} = zeros(dim); 
end
D = zeros(dim);
for i = 1:n
    Eny = diff_element(dim,directions(i));
    D   = D + Eny; 
end

%% 
iter = 0;
while iter<max_iter
    iter = iter + 1;  
    % Xk = X;
    % Ek = E;
    %% X-B
	H = zeros(dim);
    for i = 1:n
       index = directions(i);
       H = H + porder_diff_T(mu*G{index}-Gamma{index},index); 
    end
    X = real( ifftn( (fftn( mu1*(M-E)+Lambda+mu*b)+H)./(mu*(1+D)+(mu+mu1)) ) );
  
    %% G
    for i = 1:n
        index = directions(i);
        switch transform
            case 'DFT'
                [G{index},tnn_G{index}] = prox_htnn_F(porder_diff(X,index)+Gamma{index}/mu,1/(n*mu)); 
            case 'DCT'
                [G{index},tnn_G{index}] = prox_htnn_C(porder_diff(X,index)+Gamma{index}/mu,1/(n*mu));
            case 'other' 
                [G{index},tnn_G{index}] = prox_htnn_C(transform_matrices,porder_diff(X,index)+Gamma{index}/mu,1/(n*mu));
        end
    end
    
    %% Li
    L1 = update_Li(X, Y2_1, mu, lambda, '1');
    L2 = update_Li(X, Y2_2, mu, lambda, '2');
    L3 = update_Li(X, Y2_3, mu, lambda, '3');
    %% b
    b = update_b(X, L1, L2, L3, Y2_1, Y2_2, Y2_3, mu);

    %%  E-T
    E = ClosedWL1(M-X+Lambda/mu1,weightTenT*lambda/mu1,eps);
    weightTenT = 1 ./ (abs(E) + 0.01);   
    %% multipliers
    dY   = M-X-E;  
    Lambda = Lambda+mu1*dY;
    for i = 1:n
        index = directions(i); 
        Gamma{index} = Gamma{index}+mu*(porder_diff(X,index)-G{index});
    end
    mu = min(rho*mu,max_mu);    
    mu1 = min(max_mu,mu1*rho);

    %% Stop criterion
    normM = norm(M(:));
    stopC = norm(M(:)-X(:)-E(:))/normM;
    change(iter)=(stopC);
    
    if (stopC < tol)
        break;
    end         

    if detail
        if iter == 5 || mod(iter, 2) == 0
            obj = sum(cell2mat(tnn_G))/n;

            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ...
                    ', stopCriterion=' num2str(stopC)]); 
        end
    end
    
    
end
end



function Li = update_Li(B, Yi2, mu, lambda1, mode)
   
    switch mode
        case '1'
            Gi = [diff(B,1,2), zeros(size(B,1),1)];
        case '2'
            Gi = [diff(B,1,1); zeros(1,size(B,2))];
        case '3'  
            Gi = B;
    end

    A = Gi + (1/mu) * Yi2;

    [U, S, V] = svd(A, 'econ');
    S_threshold = max(S - lambda1/mu, 0);
    Li = U * S_threshold * V';
end

function b = update_b(B, L1, L2, L3, Y2_1, Y2_2, Y2_3, mu)
    
    GxT_input = gradient_x_transpose(L1 - (1/mu) * Y2_1);
    GyT_input = gradient_y_transpose(L2 - (1/mu) * Y2_2);
    Id_input  = L3 - (1/mu) * Y2_3;

    numerator = GxT_input + GyT_input + Id_input + B;
    
    % 构造系统矩阵近似，由于图像梯度算子的平方近似单位阵，因此直接简化为系数乘 I
    denominator = 3 + 1; 

    b = numerator / denominator;
end

function gxT = gradient_x_transpose(Gx)
    gxT = [-Gx(:,1), diff(Gx,1,2), Gx(:,end)];
end

function gyT = gradient_y_transpose(Gy)

    gyT = [-Gy(1,:); diff(Gy,1,1); Gy(end,:)];
end
