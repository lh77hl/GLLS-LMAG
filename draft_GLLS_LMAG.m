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

% Version 1.0 - 07/12/2024
% [1] J. Peng, Y. Wang, H. Zhang, J. Wang, and D. Meng, “Exact Decomposition of Joint Low Rankness and Local Smoothness Plus Sparse Matrices,” 
%     IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 5766-5781, 2022. 
%     https://github.com/andrew-pengjj/ctv_code
 
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
p          = 3;
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
Y2_1     = zeros(dim);
Y2_2     = zeros(dim);
Y2_3     = zeros(dim);
Lambda   = zeros(dim);
weightTenT = ones(dim);

preTnnT= 0;
NOChange_counter = 0;
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

%% start
iter = 0;
while iter<max_iter
    iter = iter + 1;  

    %% XGLb---B
	H = zeros(dim);
    for i = 1:n
       index = directions(i);
       H = H + porder_diff_T(mu*G{index}-Gamma{index},index); 
    end
    X = real( ifftn( fftn( mu*(M-E)+Lambda+H)./(mu*(1+D)) ) );
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
    L1 = update_Li(X, Y2_1, mu, lambda, '1',p);
    L2 = update_Li(X, Y2_2, mu, lambda, '2',p);
    L3 = update_Li(X, Y2_3, mu, lambda, '3',p);
    b = update_b(X, L1, L2, L3, Y2_1, Y2_2, Y2_3, mu,p);
    
    %% E---T 
    E = ClosedWL1(M-X+Lambda/mu1,weightTenT*lambda/mu1,eps);
    weightTenT = 1 ./ (abs(E) + 0.01);   
    %% multipliers
    dY   = M-X-E;  
    Lambda = Lambda+mu1*dY;
    for i = 1:n
        index = directions(i); 
        Gamma{index} = Gamma{index}+mu*(porder_diff(X,index)-G{index});
    end
    Y2_1 = Y2_1 + mu*(LMAG(X,'1',p)-L1);
    Y2_2 = Y2_2 + mu*(LMAG(X,'2',p)-L2);
    Y2_3 = Y2_3 + mu*(LMAG(X,'3',p)-L3);
    mu = min(rho*mu,max_mu);    
    mu1 = min(max_mu,mu1*rho);

    %% Stop criterion
    normM = norm(M(:));
    stopC = norm(M(:)-X(:)-E(:))/normM;
    change(iter)=(stopC);
    % NOChange_counter = sum(E(:)>0);
    if (stopC < tol) %|| (NOChange_counter == preTnnT)
        break;
    end         
    % preTnnT= NOChange_counter ;

    if detail
        if iter == 1 || mod(iter, 2) == 0
            obj = sum(cell2mat(tnn_G))/n;

            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ...
                    ', stopCriterion=' num2str(stopC)]); 
        end
    end
    
    
end
end

% update-Li
function Li = update_Li(B, Yi2, mu, lambda, direction,p)
    Gi = LMAG(B,direction,p); 
    A = Gi + (1/mu) * Yi2;
    [U,S,V] = svd(reshape(A,[size(A,1)*size(A,2),size(A,3)]),'econ'); 
    S_threshold = softthre(S,lambda/mu);
    Li = U * S_threshold * V'; 
    Li = reshape(Li,[size(A,1),size(A,2),size(A,3)]);
end

% update-b
function b = update_b(B, L1, L2, L3, Y2_1, Y2_2, Y2_3, mu,p)
    m = LMAG(L1 - (1/mu) * Y2_1,'1',p);
    m = transpose_LMAG(m, 1);
    n = LMAG(L2 - (1/mu) * Y2_2,'2',p);
    n = transpose_LMAG(n, 2);
    q = LMAG(L3 - (1/mu) * Y2_3,'3',p);
    q = transpose_LMAG(q, 3);
    numerator = m + n + q + B;
    denominator = 3 + 1; 
    b = numerator / denominator;
end

function min_abs_grad = LMAG(B,direction,p)
    switch direction
        case '1'  
            grad = diff(B, 1, 1);
            grad = cat(1, grad, zeros(1, size(B,2), size(B,3)));  
        case '2'  
            grad = diff(B, 1, 2);
            grad = cat(2, grad, zeros(size(B,1),1,size(B,3)));
        case '3'  
            grad = diff(B, 1, 3);
            grad = cat(3, grad, zeros(size(B,1), size(B,2), 1));
    end

    abs_grad = abs(grad);
    min_abs_grad = min_process(abs_grad,p);
end

function output = min_process(input_tensor, p)
    [a, b, c] = size(input_tensor);
    output = input_tensor;
    for k = 1:c
        matrix = input_tensor(:, :, k);
        for i = 1:p:a-p+1
            for j = 1:p:b-p+1
                window = matrix(i:i+p-1, j:j+p-1);
                min_values = min(window, [], 1);
                matrix(i:i+p-1, j:j+p-1) = repmat(min_values, p, 1);
            end
        end
        output(:, :, k) = matrix;
    end
end

function x = softthre(a, tau)
    x = sign(a).* max( abs(a) - tau, 0);
end

function AT = transpose_LMAG(A, dim)
    sz = size(A);             
    AT = zeros(sz, 'like', A); 

    idx_all = repmat({':'},1,ndims(A));

    idx_start = idx_all;
    idx_start{dim} = 1;
    AT(idx_start{:}) = -A(idx_start{:});

    diff_A = diff(A,1,dim); 
    idx_middle = idx_all;
    idx_middle{dim} = 2:sz(dim);
    AT(idx_middle{:}) = diff_A;
end
