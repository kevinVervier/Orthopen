# Orthopen in R
Source for reproducing results presented in '
[On Learning Matrices with Orthogonal Columns or Disjoint Supports' (Vervier et al., 2014)] (http://link.springer.com/chapter/10.1007%2F978-3-662-44845-8_18)

Orthopen is an R package to solve penalized learning tasks, assuming orthogonality between tasks support. It also allows to perform feature selection and sparsity-inducing schemes.

To install Orthopen from R, type:

```{r}
install.packages('devtools')
library(devtools)
install_github("kevinVervier/orthopen")
```

Check the [vignettes](vignettes/) for examples.
