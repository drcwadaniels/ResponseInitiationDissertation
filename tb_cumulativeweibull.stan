

// stan code for cumulative weibull model

data {

 int<lower=0> Nr; //rows
 int<lower=1> Np; //predictors
 int<lower=1> Ns; // subjects
 vector[Nr] x; //intervals
 int c[Nr]; //choices
 int s[Nr]; //subjects
 matrix[Nr,Np] P; //matrix of predictors

}

parameters {

//lambda of weibull

real alpha_lambda;
vector[Np] betas_lambda; 
vector[Ns] lambda_rme; 

//k of weibull

real alpha_k; 
vector[Np] betas_k; 
vector[Ns] k_rme; 

//mixture coef
real alpha_H;
vector[Np] betas_H;
vector[Np] H_rme;

//non-timing rate
real alpha_rate;
vector[Np] betas_rate;
vector[Np] rate_rme; 



}

transformed parameters {

//vectors for weibull function

vector<lower=0,upper=1>[Nr] wblF;
vector<lower=0,upper=1>[Nr] expF;
vector<lower=0,upper=1>[Nr] Bp_t;
vector[Nr] lambda;
vector[Nr] k;
vector[Nr] H; 
vector[Nr] rate;


//fixed effects

lambda = alpha_lambda + (P * betas_lambda);  
k = alpha_k + (P * betas_k);  
H = alpha_H + (P * betas_H);
rate = alpha_rate + (P * betas_rate);




for (i in 1:Nr)
{
//random effects
  lambda[i] = lambda[i] + lambda_rme[s[i]]; 
  k[i] = k[i] + k_rme[s[i]]; 
  H[i] = H[i] + H_rme[s[i]];
  rate[i] = rate[i] + rate_rme[s[i]];

}

//weibull function

H = exp(H)./(1+exp(H));

wblF = H.*(1-exp(-(pow(exp(log(x)-lambda),exp(k)))));
expF = (1-H).*(1-exp(-(exp(rate+log(x)))));
Bp_t = wblF+expF;

}

model {

//priors

target+= normal_lpdf(alpha_lambda|log(7),1);
target+= normal_lpdf(alpha_k|log(2),1);
target+= normal_lpdf(alpha_H|log(10),1);
target+= normal_lpdf(alpha_rate|log(0.1),1);


target+= normal_lpdf(betas_lambda|0,1);
target+= normal_lpdf(betas_k|0,1);
target+= normal_lpdf(betas_H|0,1);
target+= normal_lpdf(betas_rate|0,1);


target+= normal_lpdf(lambda_rme|0,1);
target+= normal_lpdf(k_rme|0,1);
target+= normal_lpdf(H_rme|0,1);
target+= normal_lpdf(rate_rme|0,1);


//likelihood

target+= binomial_lpmf(c|1, Bp_t); //binomial likelihood

}


