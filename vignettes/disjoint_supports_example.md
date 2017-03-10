---
title: "Disjoint supports examples"
author: "Kevin Vervier"
date: "January 31, 2017"
output: html_document
---

# Introduction

This vignette contains multiple examples for solving optimization problems with disjoint supports constraints.

### Solve a Linear regression problem with disjoint supports model

For this example, we consider the following setting for simulating data and model.

```{r}
M = 1000 # number of observations
P = 100 # number of variables
T = 6 # number of tasks
```
We start by simulating a generative model $W$, a $P\times T$ matrix that relates observations and outcomes. In this example, we assume that $W$ is a matrix with disjoint supports, meaning that $\forall i \in \lbrace1,...,P\rbrace$, if $W_{ij}\neq 0$, $W_{ij'} = 0, \forall j'\neq j$.

```{r}
# fix random seed for reproducibility
set.seed(42)

# generate model
W <- matrix(rnorm(P*T),nrow=P,ncol=T)
# generate disjoint supports
supports = sample(1:T, T, replace = FALSE)
# make sure that each column has at least one non-zero weight
if(P>T) supports = c(supports,sample(1:T, P-T, replace = TRUE))
# only keep non-zero W weights, based on disjoint supports
for(i in 1:P){
  W[i,-supports[i]] = 0
}
```

Using the model $W$, we also simulate a data set made of $M=1000$ training examples $X$. Each example is represented by $P=100$ variables, and is assessed for $T=6$ tasks in $Y$.

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
step_size=50 #might require optimization
```

Everything we need to run Orthopen is ready.

```{r}
library(orthopen)
res <- orthopen(X = X,Y= Y,lambda = lambda, verbose = 1,K=K,disjoint = TRUE, step_size=step_size)

```

# Learning with disjoint supports - benchmark study
This section gives code to reproduce results presented in '
[On Learning Matrices with Orthogonal Columns or Disjoint Supports' (Vervier et al., 2014)] (http://link.springer.com/chapter/10.1007%2F978-3-662-44845-8_18) (Fig.3).
In this experiment, we demonstrated that not convex models lead to models with higher orthogonality between columns.

### Parameters
```{r}

NVAR = 10 #number of variables
NTRAIN = seq(10,50,by=5) #training set size
SIZE = 1000 #X size
T = 10 #number of tasks
noise = 1 # gaussian noise variance

#params
LAMBDAS = 10^c(-2:0) #regularization parameters grid
DIAGVAL = unique(c(0.1,0.5,1,(T-1),2*T)) #covering non-convex, orthopen and convex cases
nrepeats = 100 

```

### Generate data using disjoint supports

```{r}
# initiate constraint matrix
K = matrix(1,nrow=T,ncol=T)
# training set
X <- matrix(rnorm(SIZE*NVAR),nrow=SIZE,ncol=NVAR) 
#standardize
X <- scale(X)
center.x = attributes(X)$'scaled:center'
scale.x = attributes(X)$'scaled:scale' 
# random model matrix
W <- matrix(rnorm(NVAR*T),nrow=NVAR,ncol=T)
# generate disjoint supports
supports = sample(1:T, NVAR, replace = TRUE)
while(length(unique(supports)) != T) supports = sample(1:T, NVAR, replace = TRUE)
for(i in 1:NVAR){
    W[i,-supports[i]] = 0
}
W[-(1:NVAR),]  = 0
# outcome variable
Y <- X %*% W + matrix(rnorm(SIZE*T),nrow=SIZE)*noise
#loop on Y columns
#for(i in 1:T){
#    #centered
#    y <- scale(Y[,i],scale=FALSE)
#    center.y = attributes(y)$'scaled:center'
#    Y[,i] = y
#}

```

### Benchmark

```{r}
#source
require(glmnet)
library(orthopen)

#loop on repeats
VALUES = 0 # initiate MSE vector
DISJOINTS = 0 # initiate disjoint support recovery vector
for(nrep in 1:nrepeats){
    print(paste("Repeat number ", nrep,sep=""))
    BEST_VALUES = NULL
    BEST_DISJOINT = NULL
    #loop on train size
    for(ntrain in NTRAIN){
        cat("ntrain: ",ntrain,"\n")
        step_size = 10^-1
        print(ntrain)
        RES = NULL
        #subset train/test sets
        train = sample(1:SIZE,ntrain)
        X.train = X[train,]
        X.test = X[-train,]
        Y.train = Y[train,]
        Y.test = Y[-train,]
        
        #5-folds cv
        folds = sample(1:5,ntrain,replace=TRUE)

        # L2-regression (ridge)
        print("ridge")
        PREDS = NULL
        MODELS = NULL
        for(model in 1:T){
            fit=glmnet(X.train,Y.train[,model],alpha=0,lambda=sort(LAMBDAS,decreasing=TRUE),intercept = FALSE)
            pred=predict(fit,X.test)
            mte=apply(0.5*(pred-Y.test[,model])^2,2,mean )
            PREDS = rbind(PREDS,mte)
            MODELS = rbind(MODELS,matrix(coef(fit)[-1,],nrow=NVAR))
        }
        glmnet.perfs = min(apply(PREDS,2,sum))
        best_lambda = which.min(apply(PREDS,2,sum))
        #get W disjointness
        W_opt = matrix(MODELS[,best_lambda],nrow=NVAR)
        tmp = 0
        for(r in 1:NVAR){
            idx = which(W_opt[r,] != 0)
            if(length(idx) == 1){
                if(supports[r] == idx) tmp = tmp + 1
            } 
        }
        test = rep(glmnet.perfs, length(LAMBDAS))
        disjointness = rep(tmp/NVAR,length(LAMBDAS))
        #update results 
        RES = rbind(RES, cbind("ridge", LAMBDAS, test, disjointness))
        
        # L1-regression (lasso)
        print("lasso")
        PREDS = NULL
        MODELS = NULL
        for(model in 1:T){
            fit=glmnet(X.train,Y.train[,model],alpha=1,intercept = FALSE,lambda=sort(LAMBDAS,decreasing=TRUE))
            pred=predict(fit,X.test)
            mte=apply(0.5*(pred-Y.test[,model])^2,2,mean )
            PREDS = rbind(PREDS,mte)
            MODELS = rbind(MODELS,matrix(coef(fit)[-1,],nrow=NVAR))
        }
        glmnet.perfs = min(apply(PREDS,2,sum))
        best_lambda = which.min(apply(PREDS,2,sum))
        #get W disjointness
        W_opt = matrix(MODELS[,best_lambda],nrow=NVAR)
        tmp = 0
        for(r in 1:NVAR){
            idx = which(W_opt[r,] != 0)
            if(length(idx) == 1){
                if(supports[r] == idx) tmp = tmp + 1
            } 
        }
        test = rep(glmnet.perfs, length(LAMBDAS))
        disjointness = rep(tmp/NVAR,length(LAMBDAS))
        #update results
        RES = rbind(RES, cbind("lasso", LAMBDAS, test, disjointness))
        
        # Orthopen (no disjoint support contraint)
        print("orthopen")
        penalty = "3" 
        for(diag_val in DIAGVAL){
            cat("diagVal: ",diag_val,"\n")
            diag(K) = diag_val
            if(diag_val<(T-1)){
                nrep_conv=5
            }else{
                nrep_conv=1
            }
            test = c()
            disjointness = c()
            for(lambda in LAMBDAS){
                print(lambda)
                #solve convex problem
                W_opt =  orthopen(X = X.train,Y= Y.train,lambda = lambda, verbose = 1,K = K, step_size = step_size, disjoint = FALSE)
                W_opt <- W_opt$W
                #MSE
                test = c(test, 0.5*sum((X.test%*%W_opt - Y.test)^2) /nrow(X.test))
                #disjointness: ratio between recovered disjoint features of learned W / true W
                tmp = 0
                for(r in 1:NVAR){
                    idx = which(W_opt[r,] != 0)
                    if(length(idx) == 1){
                        if(supports[r] == idx) tmp = tmp + 1
                    } 
                }
                disjointness = c(disjointness, tmp/NVAR)
            }
            #update res
            RES = rbind(RES, cbind(paste("Xiao_diag",diag_val,sep=""), LAMBDAS, test, disjointness))
        }
        
        # disjoint supports
        print("disjoint supports")
        for(diag_val in DIAGVAL){
            cat("diagVal: ",diag_val,"\n")
            diag(K) = diag_val
            if(diag_val<(T-1)){
                nrep_conv=5
            }else{
                nrep_conv=1
            }
            test = c()
            disjointness = c()
            for(lambda in LAMBDAS){
                print(lambda)
                step_size = 0.1
                #solve convex problem
                W_opt = orthopen(X = X.train,Y= Y.train,lambda = lambda, verbose = 1,K = K, step_size = step_size, disjoint = TRUE)
                W_opt <- W_opt$W
                #MSE
                test = c(test, 0.5*sum((X.test%*%W_opt - Y.test)^2) /nrow(X.test))
                #disjointness: ratio between recovered disjoint features of learned W / true W
                tmp = 0
                for(r in 1:NVAR){
                    idx = which(W_opt[r,] != 0)
                    if(length(idx) == 1){
                        if(supports[r] == idx) tmp = tmp + 1
                    } 
                }
                disjointness = c(disjointness, tmp/NVAR)
            }
            #add res
            RES = rbind(RES, cbind(paste("disjoint_diag",diag_val,sep=""), LAMBDAS, test, disjointness))
            
        }
        
        #get best values for each method
        best_values = by(as.numeric(RES[,3]),factor(RES[,1],levels=c("ridge","lasso",paste("Xiao_diag",DIAGVAL,sep=""),paste("disjoint_diag",DIAGVAL,sep=""))),min)
        #get corresponding disjointness values
        best_disjoint = as.numeric(c(RES[1,4],RES[1+length(LAMBDAS),4],RES[which(RES[,3] %in% best_values[-(1:2)]),4]))
        #merge results
        BEST_VALUES = cbind(BEST_VALUES,best_values)
        BEST_DISJOINT = cbind(BEST_DISJOINT,best_disjoint)
        
    }
    #BEST_VALUES
    colnames(BEST_VALUES) = NTRAIN
    rownames(BEST_VALUES) = c("ridge","lasso",paste("Xiao_diag",DIAGVAL,sep=""), paste("disjoint_diag",DIAGVAL,sep=""))
    #BEST_DISJOINT
    colnames(BEST_DISJOINT) = NTRAIN
    rownames(BEST_DISJOINT) = c("ridge","lasso",paste("Xiao_diag",DIAGVAL,sep=""), paste("disjoint_diag",DIAGVAL,sep=""))
    #paste values
    VALUES = VALUES + BEST_VALUES
    DISJOINTS = DISJOINTS + BEST_DISJOINT
    print(VALUES/nrep)

}
#mean values
VALUES = VALUES/nrepeats
DISJOINTS = DISJOINTS/nrepeats
```

### Benchmark: cross-validated mean square error

```{r}

RANGE_VALUES = range(as.numeric(VALUES))
RANGE_DISJOINT = range(as.numeric(DISJOINTS))

NAMES.legend = c("ridge","lasso",paste("xiao_diag",DIAGVAL,sep=""),paste("disjoint_diag",DIAGVAL,sep=""))
# mean square error
plot(NTRAIN,VALUES[1,],main="test error function of train size",ylim=RANGE_VALUES,col=rainbow(length(NAMES.legend))[1],type="l",lwd=2,ylab="MSE")
for(i in 2:length(NAMES.legend)){
    lines(NTRAIN,VALUES[i,],col=rainbow(length(NAMES.legend))[i],lwd=2)
} 
legend(x="topright",legend=NAMES.legend,col=rainbow(length(NAMES.legend)),pch=16)
```

### Benchmark: support recovery
```{r}
# how disjoint is the model ?
plot(NTRAIN,DISJOINTS[1,],main="disjointness function of train size",ylim=RANGE_DISJOINT,col=rainbow(length(NAMES.legend))[1],type="l",lwd=2,ylab="disjointness")
for(i in 2:length(NAMES.legend)){
    lines(NTRAIN,DISJOINTS[i,],col=rainbow(length(NAMES.legend))[i],lwd=2)
} 
legend(x="topleft",legend=NAMES.legend,col=rainbow(length(NAMES.legend)),pch=16)
```
