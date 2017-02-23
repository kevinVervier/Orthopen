#' Projection function for disjoint supports
#' @param \code{w} the weight matrix used in loss function 
#' @param \code{v} the constraint matrix used for projection
#' @export 
#' @return a list of the projected matrices \code{w} and \code{v}
#' 
proj_disjoint<-function(w,v){
  
  # store a local copy of both matrices
  w_tmp = w
  v_tmp = v
  
  # filter cases where V_ij is negative --> project on 0
  idx1 = v <= 0
  v_tmp[idx1] = 0
  w_tmp[idx1] = 0
  
  # compute once absolute value of W
  abs_w = abs(w)
  
  # get positions where no projection is needed
  idx2 = abs_w < v
  
  # get positions for which action is required
  idx3 = !(idx1 | idx2)
  # apply projection
  tmp = 0.5*(v[idx3] + abs_w[idx3])
  v_tmp[idx3] = tmp
  w_tmp[idx3] = sign(w[idx3])*tmp
  return(list("w"=w_tmp,"v"=v_tmp))
}
