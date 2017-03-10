---
title: "Orthogonal columns examples"
author: "Kevin Vervier"
date: "January 31, 2017"
output: html_document
---

### Introduction

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
set.seed(42)
res <- orthopen(X = X,Y= Y,lambda = lambda, verbose = 1,K=K,disjoint = FALSE, step_size=step_size)

set.seed(43)
res <- orthopen(X = X,Y= Y,lambda = lambda, verbose = 1,K=K,disjoint = FALSE, step_size=step_size)

```


