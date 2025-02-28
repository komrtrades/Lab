rm(list = ls())
## regression - neural networks
library(MASS)
help(Boston)

library(nnet)
dim(Boston)
# 14 variables -> 13 predictors =  input neurons
nnh<- nnet(formula = medv~.,
           data = Boston,
           trace=FALSE,
           linout = TRUE,
           size = 5, # M
           maxit=10000)
nnh
# a 13-5-1 network -> 13 predictors = input neurons

# rad is treated as a continuous predictor
table(Boston$rad)

# we can turn rad into a factor:
Boston_factor = Boston
Boston_factor$rad = factor(Boston$rad)
nlevels(Boston_factor$rad)
# 9 levels -> 8 input neurons will be used for rad
# those input neurons will be {0,1}.

dim(Boston_factor)
nnh<- nnet(formula = medv~.,
           data = Boston_factor,
           trace=FALSE,
           linout = TRUE,
           size = 5, # M
           maxit=10000)
nnh
# a 20-5-1 
# 13 predictors:
# 12 continuous -> 12 input neurons
# 1 qualitative with 9 levels -> 8 input neurons
# total = 20 input neurons.

# try nnet with 1 predictor only: rad
# continuous rad:
nnh<- nnet(formula = medv~rad,
           data = Boston,
           trace=FALSE,
           linout = TRUE,
           size = 5, # M
           maxit=10000)
nnh
# a 1-5-1 network
# 1 input neuron (continuous predictor)

# factor rad:
nnh<- nnet(formula = medv~rad,
           data = Boston_factor,
           trace=FALSE,
           linout = TRUE,
           size = 5, # M
           maxit=10000)
nnh
# a 8-5-1 network
# 8 input neurons
# rad has 9 levels -> 1 incorporated as baseline in the intercept

