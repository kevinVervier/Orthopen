---
title: "Orthogonal columns examples"
author: "Kevin Vervier"
date: "January 31, 2017"
output: html_document
---

# Introduction

This vignette contains multiple examples for solving optimization problems with orthogonal columns constraints.

### Solve a Linear regression problem with orthogonal columns

For this example, we consider the following setting for simulating data and model.

```{r}
M = 50 # number of observations
P = 10 # number of variables
T = 10 # number of tasks
```
We start by simulating a generative model $W$, a $P\times T$ matrix that relates observations and outcomes. In this example, we assume that $W$ is a matrix with orthogonal columns, meaning that $W_i^tW_j = 0, \forall i,j \in \lbrace1,...,T\rbrace$.

```{r}
# fix random seed for reproducibility
set.seed(42)

# generate model
W <- qr.Q(qr(matrix(rnorm(P*T),nrow=P,ncol=T)))

# check that columns are orthogonal:
t(W)%*%W
```

Using the model $W$, we also simulate a data set made of $M=50$ training examples $X$. Each example is represented by $P=10$ variables, and is assessed for $T=10$ tasks in $Y$.

```{r}
# simulate observations
X <- matrix(rnorm(M*P),nrow=M,ncol=P) 
# standardize
X <- scale(X)
# corresponding outcome, with gaussian noise addition
noise = 0.01
Y <- X %*% W + matrix(rnorm(M*T),nrow=M)*noise
```

The following parameters are used by Orthopen solver and need to be defined before running it.
```{r}
# orthogonality constraint matrix, where each weight represents the constraint put on a pair of columns in W
K = matrix(1,nrow=T,ncol=T)
# based on previous results, it seems that using a small diagonal value leads to more orthogonal models
diagval=0.5
diag(K) = diagval

# regularization parameter
lambda=10^-3 #might require optimization
# subgradient step size
step_size = 0.1 #might require optimization
```

Everything we need to run Orthopen is ready.

```{r}
library(orthopen)

res <- orthopen(X = X,Y= Y,lambda = lambda, verbose = 1,K=K,disjoint = FALSE, step_size=step_size)

```

# Convexity impact on model columns orthogonality

This section gives code to reproduce results presented in '
[On Learning Matrices with Orthogonal Columns or Disjoint Supports' (Vervier et al., 2014)] (http://link.springer.com/chapter/10.1007%2F978-3-662-44845-8_18) (Fig.2).
In this experiment, we demonstrated that not convex models lead to models with higher orthogonality between columns.

### Parameters
```{r}
T=10 # number of tasks
NTRAIN=50 # number of training samples
NVAR=10 # dimension
NTEST=1000 # number of test samples
step_size = 0.1 # step size for subgradient optimization
DIAGVAL <- seq(1,2*T,by=2) # diagonal values tested
LAMBDAS = 1.6^seq(-16,4) # lambda (regularization parameter) tested
ndiag <- length(DIAGVAL)
nlambda <- length(LAMBDAS)
nonconvexrep = 1 # we simply run the subgradient optimization starting from the null matrix
nrepeats = 100 # number of repeats of the experiment
K = matrix(1,nrow=T,ncol=T) # penalty matrix (we will just change the diagonal)
TEST = list() # initiate output variable
```

### Run experiments for 3 different noise levels in simulated data.

```{r}
for (noise in c(1,2.5,4)) {
  # Main loop: repeat the experiment nrepeats times
  res <- matrix(0,nrow=nlambda,ncol=ndiag) # filled at each experiment
  ang <- matrix(0,nrow=nlambda,ncol=ndiag) # filled at each experiment
  TEST = matrix(0,nrow=nrepeats,ncol= ndiag)
  ANGLES = matrix(0,nrow=nrepeats,ncol= ndiag)
  
  run_exp <- function(irep) {
    
    cat("Repeat ",irep,"\n",sep="")
    
    ## Generate random data and random model
    # X train
    X.train <- matrix(rnorm(NTRAIN*NVAR),nrow=NTRAIN,ncol=NVAR)
    # X test
    X.test <- matrix(rnorm(NTEST*NVAR),nrow= NTEST,ncol=NVAR)
    # Random orthogonal matrix
    W <- qr.Q(qr(matrix(rnorm(NVAR*T),nrow=NVAR,ncol=T)))
    # Y train
    Y.train <- X.train %*% W + matrix(rnorm(NTRAIN*T),nrow=NTRAIN)*noise
    # Y test (without noise)
    Y.test <- X.test %*% W
    
    # Loop on the diagonal value of K
    for(idiag in seq(length(DIAGVAL))){
      cat('diag=',DIAGVAL[idiag])
      diag(K) = DIAGVAL[idiag]
      
      # Loop on regularization parameter lambda
      for(ilambda in seq(LAMBDAS)){
        cat('.')
        
        # Train predictor
        W_opt = orthopen(X = X.train,Y= Y.train,lambda = LAMBDAS[ilambda], verbose = 0, K = K, step_size = step_size,max_iter=1e5)
        
        # Performance (MSE) on the test set
        res[ilambda,idiag] <- 0.5*sum((X.test%*%W_opt$W - Y.test)^2) /NTEST
        
        # Angles between columns of the predictor
        v <- acos(cor(W_opt$W))*360/(2*pi) - 90
        diag(v) <- 0
        ang[ilambda,idiag] <- sum(abs(v))/(T*(T-1))
        
      } # end of ilambda loop
      cat('\n')
    } # end of idiag loop
    
    # Find the best lambda for each diag
    bb <- apply(res,2,which.min)
    return(list(sapply(seq(ndiag),function(i){res[bb[i],i]}), sapply(seq(ndiag),function(i){ang[bb[i],i]}))) #return 
    
  }
  # for each noise level, run 'nrepeats' experiments
  TEST[[noise]] <- lapply(seq(nrepeats),run_exp)
}

```

### Plot results - Mean Square Error as a function of convexity

```{r}
#graphical parameters
par(oma=c(2,2,0,0))
par(mar=c(3,3,1,0.5))
par(mfrow=c(1,3))

# get results for noise level 1: mean value + standard error
MSE <- lapply(TEST[[noise[1]]],function(u){u[[1]]}) # get results for noise level 1: mean value + standard error
mres1 <- apply(as.data.frame(MSE),1,mean)
sres1 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres1,main="",ylim=range(c(mres1+sres1,mres1-sres1)),type="l",lwd=4,ylab="",xlab='',pch=16,cex.axis=1.5)
errbar(DIAGVAL,mres1,mres1+sres1,mres1-sres1,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

# get results for noise level 2: mean value + standard error
MSE <- lapply(TEST[[noise[2]]],function(u){u[[1]]})
mres2 <- apply(as.data.frame(MSE),1,mean)
sres2 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres2,main="",ylim=range(c(mres2+sres2,mres2-sres2)),type="l",lwd=4,ylab="",xlab='',pch=26,cex.axis=1.5)
errbar(DIAGVAL,mres2,mres2+sres2,mres2-sres2,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

# get results for noise level 3: mean value + standard error
MSE <- lapply(TEST[[noise[3]]],function(u){u[[1]]})
mres3 <- apply(as.data.frame(MSE),1,mean)
sres3 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres3,main="",ylim=range(c(mres3+sres3,mres3-sres3)),type="l",lwd=4,ylab="",xlab='',pch=16,cex.axis=1.5)
errbar(DIAGVAL,mres3,mres3+sres3,mres3-sres3,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

mtext("Diagonal",side=1,outer=TRUE,cex=1.5)
mtext("Test MSE",side=2,outer=TRUE,cex=1.5)

```

### Plot results - Mean Angle between pairs of columns as a function of convexity

```{r}
#graphical parameters
par(oma=c(2,2,0,0))
par(mar=c(3,3,1,0.5))
par(mfrow=c(1,3))

# get results for noise level 1: mean value + standard error
MSE <- lapply(TEST[[noise[1]]],function(u){u[[2]]})
mres1 <- apply(as.data.frame(MSE),1,mean)
sres1 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres1,main="",ylim=range(c(mres1+sres1,mres1-sres1)),type="l",lwd=4,ylab="",xlab='',pch=16,cex.axis=1.5)
errbar(DIAGVAL,mres1,mres1+sres1,mres1-sres1,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

# get results for noise level 2: mean value + standard error
MSE <- lapply(TEST[[noise[2]]],function(u){u[[2]]})
mres2 <- apply(as.data.frame(MSE),1,mean)
sres2 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres2,main="",ylim=range(c(mres2+sres2,mres2-sres2)),type="l",lwd=4,ylab="",xlab='',pch=26,cex.axis=1.5)
errbar(DIAGVAL,mres2,mres2+sres2,mres2-sres2,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

# get results for noise level 3: mean value + standard error
MSE <- lapply(TEST[[noise[3]]],function(u){u[[2]]})
mres3 <- apply(as.data.frame(MSE),1,mean)
sres3 <- apply(as.data.frame(MSE),1,sd)*1.96/sqrt(nrepeats)
plot(DIAGVAL,mres3,main="",ylim=range(c(mres3+sres3,mres3-sres3)),type="l",lwd=4,ylab="",xlab='',pch=16,cex.axis=1.5)
errbar(DIAGVAL,mres3,mres3+sres3,mres3-sres3,add=T,pch=1,cap=0.01)
grid()
abline(v=ncol(K)-1,lty = 2,lwd=2)

mtext("Diagonal",side=1,outer=TRUE,cex=1.5)
mtext("mean(|angle-90|)",side=2,outer=TRUE,cex=1.5)
  
```

