################################################################
### Main function that solves a general problem (primal LS): ###
###                                ###
################################################################

#' Solver for a general problem of the form: min (loss + Orthopen)  
#' 
#' \code{orthopen}
#' @param X \eqn{MxP} observations matrix (features are in columns)
#' @param Y \eqn{MxT} observed output matrix for T different tasks
#' @param lambda a regularization parameter (default \code{1})
#' @param step_size step for gradient descent (default \code{0.1})
#' @param verbose option (default \code{0})
#' @param stop_no_improve number of gradient descent steps without improvment before stopping (default: \code{100})
#' @param max_iter  maximum number of iterations before stopping optimization (default: \code{1000000})
#' @param K \eqn{PxP} orthogonality constraints matrix (default: diagonal matrix --> no orthogonality constraint)
#' @param disjoint if \code{TRUE} add contraints for disjoint supports (default: TRUE)
#' @param logistic if \code{TRUE} change loss function to logistic loss (for classification problems)
#' @param enet if \code{TRUE}, add a \code{L1} penalization to the \code{L2} penalization, using elastic net formula (single parameter lambda, assuming \eqn{enet = 0.5*L2 + 0.5*L1})
#' @return a list containing three elements\itemize{\item \code{W}: optimal \eqn{PxT} matrix for objective function minimum \item obj: objective function value at W \item imax: number of steps before reaching optimum
#' }
#' @export
#' @examples #solve orthogonal columns problem 
#' # min_W 1/2 norm( X%*%W - Y )^2 + lambda ||W||_orthopen
#' NVAR=10
#' NTRAIN=100
#' T=3
#' K = matrix(1,nrow=T,ncol=T)
#' # Generate random data and random model
#' X <- matrix(rnorm(NTRAIN*NVAR),nrow=NTRAIN,ncol=NVAR)
#' # Random orthogonal matrix
#' W <- qr.Q(qr(matrix(rnorm(NVAR*T),nrow=NVAR,ncol=T)))
#' Y <- X %*% W + matrix(rnorm(NTRAIN*T),nrow=NTRAIN)
#' set.seed(42)
#' res <- main(X,Y,lambda = 0.1,K = K,disjoint = FALSE)


main <- function(X,Y,lambda=1, step_size=0.1, verbose = 0, stop_no_improve=100, max_iter=1e6, K=NULL,disjoint=TRUE,logistic = FALSE,enet= FALSE){
  
  #get problem dimensions
  m = nrow(X)
  p = ncol(X)
  T = ncol(Y)
  
  ############
  ### INIT ###
  ############
  #init K if not provided
  if(is.null(K)) K = diag(1,T)
  #init W 
  if(!disjoint){
    W_k = matrix(0,nrow=p,ncol=T)
  }else{
    # limit impact of W_0 on disjoint support problem
    W_k = matrix(rnorm(p*T),ncol=T)
    W_k = abs(W_k)
  }
  #store best reached point
  W = W_k
  #define V matrices if disjoint supports are needed
  if(disjoint){
    V_k = W_k
    V <- V_k
  }
  
  #derive L1-regularization parameters if enet 
  if(enet) lambda1 = 0.5*diag(K)*lambda
  
  #objective function value and number non-improved steps, used in stop criterion
  new <- Inf
  no_improv = 0
  
  #steps counter, used in gradient step_size
  i = 0
  #################
  ### MAIN LOOP ###
  #################
  
  if(verbose >0) cat('Step \t ObjFun\t NonImproving\n',sep='')
  
  # descent until reaching non-improvment plateau OR max_iter (if max_iter is reached, you may want to change step_size parameter for instance)
  while((no_improv < stop_no_improve) && (i<max_iter)){
    
    # get sparse support size (only for verbose purpose)
    if(verbose >0) idx = which(W_k != 0,arr.ind = TRUE)
    
    #increment i
    i <- i+1
    #verbose on current state
    if(verbose >0 & i%%1000 == 0 ){ 
      cat(i,'\t',new,'\t',no_improv,'\n',sep='')
      cat('Current sparse support:',length(W_k)-nrow(idx),'\n')
    }
    
    # compute matrix products once
    if(!logistic){
      LS = X%*%W_k - Y
    }else{
      LS = log(1 + exp(-Y*X%*%W_k))
    } 
    scale = sqrt(i)
    if(disjoint){
      PEN = crossprod(V_k)#t(V_k)%*%V_k
      if(enet) M = V_k
    }else{
      PEN = crossprod(W_k)#t(W_k)%*%W_k
      if(enet) M = W_k
    }
    # get current objective function value
    if(!logistic){ 
      if(enet){
        tmp = 0.5* (sum((LS)^2)/nrow(X) + 0.5*lambda * sum(abs(PEN) * K)) + 0.5*lambda*sum(abs(t(M))*diag(K))
      }else{
        tmp = 0.5* (sum((LS)^2)/nrow(X) + lambda * sum(abs(PEN) * K))
      }
    }else{
      if(enet){
        tmp = sum(LS)/nrow(X) + 0.5*lambda * sum(abs(PEN) * K) + 0.5*lambda*sum(abs(t(M))*diag(K))
      }else{
        tmp = sum(LS)/nrow(X) + 0.5*lambda * sum(abs(PEN) * K) 
      }
    }
    
    #subgradient scheme is not a strict descent scheme
    if(tmp < new && abs(tmp-new) > 10^-5){
      no_improv <- 0
      new <- tmp
      W <- W_k
      if(disjoint) V <- V_k
    }else{
      no_improv <- no_improv + 1
    }
    
    # Get subgradient at current W_k (and V_k, if disjoint support)      
    if(disjoint){
      if(!logistic){
        gradientW <- crossprod(X,LS)/nrow(X)
      }else{
        gradientW <- -crossprod(X,Y - 1/(1+exp(-X%*%W_k)))/nrow(X)
      }
      gradientV <- lambda * V_k%*%(sign(PEN)*K)
      if(enet) gradientV = 0.5*gradientV + 0.5*lambda*sign(V_k)
      
      norm_gradient <- sqrt(sum((gradientW + gradientV)^2))
    }else{
      if(!logistic){
        if(enet){
          gradientW <- crossprod(X,LS)/nrow(X) + 0.5*lambda * W_k%*%(sign(PEN)*K) + 0.5*lambda*sign(W_k)
        }else{
          gradientW <- crossprod(X,LS)/nrow(X) + lambda * W_k%*%(sign(PEN)*K)
        }
        
      }else{
        if(enet){
          gradientW <- -crossprod(X,Y - 1/(1+exp(-X%*%W_k)))/nrow(X) + lambda * W_k%*%(sign(PEN)*K) + 0.5*lambda*sign(W_k)
        }else{
          gradientW <- -crossprod(X,Y - 1/(1+exp(-X%*%W_k)))/nrow(X) + lambda * W_k%*%(sign(PEN)*K) 
        }
        
      }
      norm_gradient <- sqrt(sum((gradientW)^2))
    }
    
    # apply subgradient 'descent' on current W_k and V_k
    if (norm_gradient>0) {
      scale_grad = scale*norm_gradient
      # Subgradient update
      W_k <- W_k - step_size*gradientW/scale_grad
      if(disjoint) V_k <- V_k - step_size*gradientV/scale_grad
      
      #projection step
      if(disjoint){
        projection <- proj_disjoint(w = W_k, v = V_k)
        W_k <- projection$w
        V_k <- projection$v
      }
    } else {
      # Stop: we have found a (perhaps local) minimum
      no_improv <- stop_no_improve+1
    }
  }
  #store i_max: step of w*
  imax = i
  ##############
  ### OUTPUT ###
  ##############
  best_obj <- new
  w_star <- W 
  return(list(W=w_star,obj=new,imax=imax))
}