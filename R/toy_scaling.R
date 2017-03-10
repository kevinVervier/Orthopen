#####################
# Disjoint Supports #
#####################

nvar = 8000
# number of training points
nobs = 1300
# number of classes
task = 6
#
diagval=0.5
#toy dataset
K = matrix(1,nrow=task,ncol=task)
diag(K) = diagval
# regularization parameter
lambda=10^-3
# noise level for simulated data
noise = 0.01
# random seed for reproducibility
set.seed(42)

# generate model
W <- matrix(rnorm(nvar*task),nrow=nvar,ncol=task)
#disjoint supports
supports = sample(1:task, nvar, replace = TRUE)
while(length(unique(supports)) != task) supports = sample(1:task, nvar, replace = TRUE)

for(i in 1:nvar){
  W[i,-supports[i]] = 0
}
# observations
X <- matrix(rnorm(nobs*nvar),nrow=nobs,ncol=nvar) 
#standardize
X <- scale(X)
center.x = attributes(X)$'scaled:center'
scale.x = attributes(X)$'scaled:scale' 
# corresponding outcome
Y <- X %*% W + matrix(rnorm(nobs*task),nrow=nobs)*noise

source("orthopen.R")
set.seed(42)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50))

set.seed(43)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50))

#######################
# logistic regression #
#######################

# time profiling of one run
nvar = 8000
# number of training points
nobs = 1300
# number of classes
task = 6
#
diagval=0.5
#toy dataset
K = matrix(1,nrow=task,ncol=task)
diag(K) = diagval
# regularization parameter
lambda=10^-3
# noise level for simulated data
noise = 0.01

# random seed for reproducibility
set.seed(42)

# generate model
W <- matrix(rnorm(nvar*task),nrow=nvar,ncol=task)
#disjoint supports
supports = sample(1:task, nvar, replace = TRUE)
while(length(unique(supports)) != task) supports = sample(1:task, nvar, replace = TRUE)

for(i in 1:nvar){
  W[i,-supports[i]] = 0
}
# observations
X <- matrix(rnorm(nobs*nvar),nrow=nobs,ncol=nvar) 
#standardize
X <- scale(X)
center.x = attributes(X)$'scaled:center'
scale.x = attributes(X)$'scaled:scale' 
# corresponding outcome
Y <- sign(X %*% W + matrix(rnorm(nobs*task),nrow=nobs)*noise)

source("main.R")
set.seed(42)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50,logistic=TRUE))

set.seed(43)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50,logistic=TRUE))

####################
# Test Elastic-Net #
####################  
nvar = 800
# number of training points
nobs = 130
# number of classes
task = 6
#
diagval=0.5
#toy dataset
K = matrix(1,nrow=task,ncol=task)
diag(K) = diagval
# regularization parameter
lambda=10^-3
# noise level for simulated data
noise = 0.01

# random seed for reproducibility
set.seed(42)

# generate model
W <- matrix(rnorm(nvar*task),nrow=nvar,ncol=task)
#disjoint supports
supports = sample(1:task, nvar, replace = TRUE)
while(length(unique(supports)) != task) supports = sample(1:task, nvar, replace = TRUE)

for(i in 1:nvar){
  W[i,-supports[i]] = 0
}
# observations
X <- matrix(rnorm(nobs*nvar),nrow=nobs,ncol=nvar) 
#standardize
X <- scale(X)
center.x = attributes(X)$'scaled:center'
scale.x = attributes(X)$'scaled:scale' 
# corresponding outcome
Y <- sign(X %*% W + matrix(rnorm(nobs*task),nrow=nobs)*noise)


source("main.R")
set.seed(42)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50,enet=TRUE))

set.seed(43)
system.time(res <- main(X = X,Y= Y,lambda = lambda, verbose = 1,K = K,disjoint = TRUE, step_size=50,enet=TRUE))
