function [f0]=PE_BSBL_Cluster(y,fs,minPitch,maxPitch,GridInterval,Number_source)
% PE-BSBL-Cluster   Multi-pitch estimator based on Block Sparse Bayesian
% Learning with intra-block clustering
%
% Syntax:
%   f0=PE_BSBL_Cluster(y,fs,minPitch,maxPitch,GridInterval,Number_source);
%
%
% Input:
%   y               -->  multi-/single-pitch input signal (assumed complex)
%   fs              -->  sampling rate
%   minPitch        -->  the allowed minimum pitch
%   maxPitch        -->  the allowed minimum pitch
%   GridInterval    -->  the grid interval
%   Number_source   -->  the total number of sources
%
% Output:
%   f0       fundamental frequency estimate in Hz
%
% Description:
%   Estimates the fundamental frequencies of a single- or multi-pitch
%   complex signal given the total number of sources based on
%   the Block Sparse Bayesian Learning with intra-block clustering.
% Reference:
%   Liming Shi, Jesper Rindom Jensen, Jesper Kjaer Nielsen, and Mads, Graesboll Christensen
%   multipitch estimation using block sparse Bayesian learning and
%   intra-block clustering, submitted to the ICASSP 2018
% Example:
%   f0=pitch_anls(y,8000,50,500,2,2);
%
% Implemented By:
%   Liming Shi (ls@create.aau.dk)?Aalborg University, Denmark 
%
% ----------------------------
% Dictionary generation
% ----------------------------
N=length(y);
Pitch_grid=(minPitch:GridInterval:maxPitch)/fs;
P=length(Pitch_grid);
Lmax=10;
Z = CreateDictionary(N, Lmax,Pitch_grid);
% ----------------------------
% data normalization
% ----------------------------
y_per = 1/sqrt(N)*abs(fft(y,fs));
y = y/max(y_per);
Z=Z/sqrt(N);
% ----------------------------
% design parameters
% ----------------------------
c=1e-6;
d=1e-6;
g=1;
h=1e-6;

k=1;m1=Lmax;
e_rest=[k/m1,1,k/m1,(1-1/m1)*k];
f_rest=[(1-1/m1)*k,1e6,k/m1,k/m1];
e_1_Lmax=[1/Lmax,1-1/Lmax];
f_1_Lmax=[1-1/Lmax,1/Lmax];

% ----------------------------
% iteration number setting
% ----------------------------
max_iters=1000;
% --------------------------------------------------------
% Initialization (statistics and parameter settings)
% --------------------------------------------------------
% indicator \theta_p ~ Bernoulli(\pi_p)
mu_theta=ones(P*Lmax,1);
for ii=1:P*Lmax
    if rem(ii,Lmax)~=0                
         mu_theta(ii)=1-rem(ii,Lmax)/(Lmax+1);
    else
        mu_theta(ii)=1-Lmax/(Lmax+1);
    end
end
% noise precision alpha0~ Gamma(c,d)
mu_gamma=1e2/std(y)^2;

% complex_amplitude precision \alpha_p~ Gamma(g,h)
Zs=(Z'*y);
ratio = abs(Zs).^2;
ratio1=mean(reshape(ratio,Lmax,P),1);
mu_alpha=1./abs(ratio1(:)-1/mu_gamma);

% success prabability \pi ~ Beta(e,f)
mu_log_PI=ones(Lmax,1);
mu_log_one_minus_PI=ones(Lmax,1);
for ii=1:Lmax
    if rem(ii,Lmax)~=0           
        mu_log_PI(ii)=(psi(mu_theta(ii)*2)-psi(2));
        mu_log_one_minus_PI(ii)=(psi((1-mu_theta(ii))*2)-psi(2));
    else
        mu_log_PI(ii)=(psi(mu_theta(ii)*2)-psi(2));
        mu_log_one_minus_PI(ii)=(psi((1-mu_theta(ii))*2)-psi(2));
    end
end
mu_log_PI=repmat(mu_log_PI,1,P);mu_log_PI=mu_log_PI(:);
mu_log_one_minus_PI=repmat(mu_log_one_minus_PI,1,P);mu_log_one_minus_PI=mu_log_one_minus_PI(:);


%-----------------------------
% complex amplitude u~ComplexGaussian(c,d)
%-----------------------------
indx_set=reshape(1:P*Lmax,Lmax,P);
ZHZ_total=Z'*Z;
clear ii; clear indx;

lamda_temp=diag(1./(kron(mu_alpha,ones(Lmax,1))+mu_gamma*diag(ZHZ_total).*mu_theta.*(1-mu_theta)));
% following is the same as Z*diag(mu_theta) but faster
Z_diag_theta=Z*diag(mu_theta);  
Z_theta_lambda=Z_diag_theta*lamda_temp;
sigma_u=lamda_temp-Z_theta_lambda'/(1/mu_gamma*eye(N)+Z_theta_lambda*Z_diag_theta')*Z_theta_lambda;
% the following is the same as mu_gamma*sigma_u*diag(mu_theta)*Zs; but faster
mu_u=mu_gamma*sigma_u*(mu_theta.*Zs);
% -----------------------------
% Variational Bayesian
% -----------------------------
PRUNE_GAMMA=1e-3;
keep_list = [1:P]';
% index_set=
for iter=1:max_iters
    %=========== Prune weights as their hyperparameters go to infinite==============
    mu_alpha_inverse=1./mu_alpha;
    if iter>=5 && (min(mu_alpha_inverse) < PRUNE_GAMMA)
        index = find(mu_alpha_inverse > PRUNE_GAMMA);
%         index_remove = find(mu_alpha_inverse <= PRUNE_GAMMA);
        if length(index)<Number_source
            [temp_value,ind_temp]=sort(mu_alpha);            
            index=ind_temp(1:Number_source);
        end
%         % prune gamma and associated components in Sigma0, Phi
        mu_alpha = mu_alpha(index);  
        
        idx=indx_set(:,index);idx=idx(:);
%         idx_remove=indx_set(:,index_remove);idx_remove=idx_remove(:);
        Z = Z(:,idx);
        mu_u=mu_u(idx);
        sigma_u=sigma_u(idx,idx);
        mu_theta=mu_theta(idx);
        mu_log_PI=mu_log_PI(idx);
        mu_log_one_minus_PI=mu_log_one_minus_PI(idx);
%         ZHZ_total1=ZHZ_total-Z(:,idx_remove)'*Z(:,idx_remove);
        ZHZ_total=Z'*Z;        
        keep_list = keep_list(index);
    end

    % (1) Update Indicator \theta statistics
    res=y-Z*(mu_u.*mu_theta);
    for j=1:size(Z,2)
        res=res+mu_theta(j)*Z(:,j)*mu_u(j);
        t1=mu_log_PI(j)-real(mu_gamma*(Z(:,j)'*Z(:,j)*...
            (mu_u(j)'*mu_u(j)+sigma_u(j,j)) - ...
            2*real(mu_u(j)'*(Z(:,j)'*res))));
        t2=mu_log_one_minus_PI(j);
        mu_theta(j) = 1/(1+exp(t2-t1));
        res=res-mu_theta(j)*Z(:,j)*mu_u(j);
    end
    
    % (4) Update  precision \alpha_p~ Gamma(g,h)
    g_tilde=g+Lmax;
    h_tilde=zeros(size(Z,2)/Lmax,1);
    for j=1:size(Z,2)/Lmax
        h_tilde(j,1)=h+real(trace(mu_u(indx_set(:,j))*mu_u(indx_set(:,j))'+sigma_u(indx_set(:,j),indx_set(:,j))));
    end
    alpha_old=mu_alpha;
    
    mu_alpha=g_tilde./h_tilde;
    
%   (3) Update noise precision alpha0~ Gamma(c,d)
    c_tilde=c+N;
    mu_thetaThetaT=mu_theta*mu_theta'+diag(mu_theta.*(1-mu_theta));
    R_tilde=(mu_u*mu_u'+sigma_u).*mu_thetaThetaT;
    mu_tilde=mu_u.*mu_theta;
%   the following is the same as  E=res'*res+trace(ZHZ_total*(R_tilde-mu_tilde*mu_tilde'));
     E=res'*res+sum(sum((ZHZ_total).*(R_tilde-mu_tilde*mu_tilde')));
    d_tilde=d+real(E);
    mu_gamma=c_tilde/d_tilde;
    


    % (2) Update complex amplitude \mathbf{u}_p ~ CN(mu,sigmaW) statistics
    if N<size(Z,2)
%         tic
        lamda_temp=diag(1./(kron(mu_alpha,ones(Lmax,1))+mu_gamma*diag(ZHZ_total).*mu_theta.*(1-mu_theta)));
        % following is the same as Z*diag(mu_theta) but faster (Z.'.*(mu_theta)).'
        Z_diag_theta=Z*diag(mu_theta);
        Z_theta_lambda=Z_diag_theta*lamda_temp;
        sigma_u=lamda_temp-Z_theta_lambda'/(1/mu_gamma*eye(N)+Z_theta_lambda*Z_diag_theta')*Z_theta_lambda;
        % the following is the same as mu_gamma*sigma_u*diag(mu_theta)*Zs; but faster
        mu_u=mu_gamma*sigma_u*(mu_theta.*(Z'*y));
%         toc
    else
%         tic
        sigma_u=inv(diag(kron(mu_alpha,ones(Lmax,1)))+mu_gamma*(ZHZ_total.*mu_thetaThetaT));
        mu_u=mu_gamma*sigma_u*(mu_theta.*(Z'*y));
%         toc
    end



    % (5) Update  success prabability \pi ~ Beta(e,f)
    P_temp=size(Z,2)/Lmax;
    tilde_e=zeros(P_temp*Lmax,4);
    tilde_f=zeros(P_temp*Lmax,4);
    mu_log_PI_pattern=zeros(P_temp*Lmax,4);
    mu_log_one_minus_PI_pattern=zeros(P_temp*Lmax,4);
    p_pattern=zeros(P_temp*Lmax,4);
    for j=1:P_temp*Lmax
        switch rem(j,Lmax)
            case 1
                p_pattern(j,1)=(1-(mu_theta(j+1)));
                tilde_e(j,1)=e_1_Lmax(1)+p_pattern(j,1)*(mu_theta(j));
                tilde_f(j,1)=f_1_Lmax(1)+p_pattern(j,1)*(1-mu_theta(j));
                mu_log_PI_pattern(j,1)=psi(tilde_e(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
                mu_log_one_minus_PI_pattern(j,1)=psi(tilde_f(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
                
                p_pattern(j,2)=mu_theta(j+1);
                tilde_e(j,2)=e_1_Lmax(2)+p_pattern(j,2)*(mu_theta(j));
                tilde_f(j,2)=f_1_Lmax(2)+p_pattern(j,2)*(1-mu_theta(j));
                mu_log_PI_pattern(j,2)=psi(tilde_e(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
                mu_log_one_minus_PI_pattern(j,2)=psi(tilde_f(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
            case 0
                p_pattern(j,1)=mu_theta(j-Lmax+1)*(1-(mu_theta(j-1)));
                tilde_e(j,1)=e_1_Lmax(1)+p_pattern(j,1)*(mu_theta(j));
                tilde_f(j,1)=f_1_Lmax(1)+p_pattern(j,1)*(1-mu_theta(j));
                mu_log_PI_pattern(j,1)=psi(tilde_e(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
                mu_log_one_minus_PI_pattern(j,1)=psi(tilde_f(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
                
                p_pattern(j,2)=1-mu_theta(j-Lmax+1);
                tilde_e(j,2)=e_rest(2)+p_pattern(j,2)*(mu_theta(j));
                tilde_f(j,2)=f_rest(2)+p_pattern(j,2)*(1-mu_theta(j));
                mu_log_PI_pattern(j,2)=psi(tilde_e(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
                mu_log_one_minus_PI_pattern(j,2)=psi(tilde_f(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
                
                p_pattern(j,3)=mu_theta(j-Lmax+1)*mu_theta(j-1);
                tilde_e(j,3)=e_1_Lmax(2)+p_pattern(j,3)*(mu_theta(j));
                tilde_f(j,3)=f_1_Lmax(2)+p_pattern(j,3)*(1-mu_theta(j));
                mu_log_PI_pattern(j,3)=psi(tilde_e(j,3))-psi( tilde_e(j,3)+tilde_f(j,3));
                mu_log_one_minus_PI_pattern(j,3)=psi(tilde_f(j,3))-psi( tilde_e(j,3)+tilde_f(j,3));
                
            otherwise                                    
                p_pattern(j,2)=1-mu_theta(fix(j/Lmax)*Lmax+1);
                tilde_e(j,2)=e_rest(2)+p_pattern(j,2)*(mu_theta(j));
                tilde_f(j,2)=f_rest(2)+p_pattern(j,2)*(1-mu_theta(j));
                mu_log_PI_pattern(j,2)=psi(tilde_e(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
                mu_log_one_minus_PI_pattern(j,2)=psi(tilde_f(j,2))-psi( tilde_e(j,2)+tilde_f(j,2));
                
                p_pattern(j,1)=mu_theta(fix(j/Lmax)*Lmax+1)*(1-(mu_theta(j-1)))*(1-(mu_theta(j+1)));
                tilde_e(j,1)=e_rest(1)+p_pattern(j,1)*(mu_theta(j));
                tilde_f(j,1)=f_rest(1)+p_pattern(j,1)*(1-mu_theta(j));
                mu_log_PI_pattern(j,1)=psi(tilde_e(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
                mu_log_one_minus_PI_pattern(j,1)=psi(tilde_f(j,1))-psi( tilde_e(j,1)+tilde_f(j,1));
              
                
                p_pattern(j,3)=mu_theta(fix(j/Lmax)*Lmax+1)*((1-mu_theta(j-1))*mu_theta(j+1)+(mu_theta(j-1))*(1-mu_theta(j+1)));
                tilde_e(j,3)=e_rest(3)+p_pattern(j,3)*(mu_theta(j));
                tilde_f(j,3)=f_rest(3)+p_pattern(j,3)*(1-mu_theta(j));
                mu_log_PI_pattern(j,3)=psi(tilde_e(j,3))-psi( tilde_e(j,3)+tilde_f(j,3));
                mu_log_one_minus_PI_pattern(j,3)=psi(tilde_f(j,3))-psi( tilde_e(j,3)+tilde_f(j,3));
                
                p_pattern(j,4)=mu_theta(fix(j/Lmax)*Lmax+1)*mu_theta(j-1)*mu_theta(j+1);
                tilde_e(j,4)=e_rest(4)+p_pattern(j,4)*(mu_theta(j));
                tilde_f(j,4)=f_rest(4)+p_pattern(j,4)*(1-mu_theta(j));
                mu_log_PI_pattern(j,4)=psi(tilde_e(j,4))-psi(tilde_e(j,4)+tilde_f(j,4));
                mu_log_one_minus_PI_pattern(j,4)=psi(tilde_f(j,4))-psi(tilde_e(j,4)+tilde_f(j,4));
        end
    end
    mu_log_PI=sum(mu_log_PI_pattern.*p_pattern,2);
    mu_log_one_minus_PI=sum(mu_log_one_minus_PI_pattern.*p_pattern,2);

   if  norm(mu_alpha-alpha_old)/norm(alpha_old)<1e-3 || iter >= max_iters   
      
       break;
   end

end
m_a=(mu_u.*mu_theta);

idx=indx_set(:,keep_list);idx=idx(:);
m_a_final=zeros(P*Lmax,1);m_a_final(idx)=m_a;
m_theta_final=zeros(P*Lmax,1);m_theta_final(idx)=mu_theta;
Cov_stat_vec=zeros(P*Lmax,1);Cov_stat_vec(idx)=real(diag((R_tilde-mu_tilde*mu_tilde')));
%% obtain estimate of pitch
matrix_result=reshape(abs(m_a_final).^2+Cov_stat_vec,Lmax,P);
BlockSpectraOut=sum(matrix_result,1);
% .*sum(reshape(m_theta_final,Lmax,P));
[~,l_adapt]=sort(BlockSpectraOut);
l_adapt=l_adapt(end-Number_source+1:end);
[l_adapt]=sort(l_adapt);
f0=Pitch_grid(l_adapt)*fs;
return

function [W]=CreateDictionary(DataLength, MaxHarms,PitchGrid)%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
NoPitches=length(PitchGrid);
W = [];
for i = 1:NoPitches
    for j=1:MaxHarms
    W = [W exp(1i*PitchGrid(i)*j*2*pi*[0:DataLength-1]')];%/sqrt(DataLength)];
    end
end

