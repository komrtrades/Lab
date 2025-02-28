# Regression Trees #####
rm(list =ls())

# load the function to compute the miscassification rate:
source("data/misc.R")

# Load the data
library(ElemStatLearn)
data("prostate")
names(prostate)
# load("data/prostate.RData")
head(prostate)

# 9 variables + a test/tran id
# remove the "train" id (last column):
x<-prostate[,-ncol(prostate)]
p<-ncol(x)-1 #no of features
n<-nrow(x) # no of units
# what happens if we don't remove "train" id?
# it will be used as a predictor!

## Grow the regression tree ####
library(tree)
head(x)
# y = lpsa
# x = the 8 variables before
set.seed(123)
tree.prostate<-tree(lpsa~.,x)
summary(tree.prostate)

tree.prostate
plot(tree.prostate) #plot the tree
text(tree.prostate,pretty=0,digits=3)
# 9 leaves in total

prune.prostate_CV = prune.tree(tree.prostate,
                               best = 9)

## Cross-validation to determinig optimal pruning ####
?cv.tree
set.seed(1234)
cvtree.prostate<-cv.tree(tree.prostate,
                         K=5, FUN=prune.tree)
cvtree.prostate

plot(y = cvtree.prostate$dev, x = cvtree.prostate$size)

#############################################
# we code the CV on our own: 
#############################################
# split the data (n observations) into K = 5 folds
n/5
19 * 3 + 40
# choose the group of each observation
group = c(rep(1:3,19), 
          rep(4:5,20))
table(group)
length(group); n

# we sample the group index at random
# why?
set.seed(1234)
index <- sample(group, 
                length(group), 
                replace=FALSE) #folds
# because observations could be ordered
# not randomly in the dataset

MSE = matrix(NA, nrow = 5, ncol = 9)
colnames(MSE) = 1:9

# loop on the number of leaves (1:7):
for(k in 2:9){
  # loop on the 5 groups:
  for(i in 1:5){
    # build and prune the tree on all data, except group "i"
    tree.prostate_CV = tree(lpsa~., 
                            x[index != i, ])
    
    prune.prostate_CV = prune.tree(tree.prostate_CV,
                                   best = k)
    
    # predict group "i"
    y_hat_CV = predict(prune.prostate_CV, x[index == i,])
    MSE[i,k] = mean( (y_hat_CV - x$lpsa[index == i])^2 )
  }
}

#Warning messages:
# 1: In prune.tree(tree.prostate_CV, best = k) :
# best is bigger than tree size
# 2: In prune.tree(tree.prostate_CV, best = k) :
#  best is bigger than tree size

# why ?
# we cannot trust values of 8 and 9.
# also, 6 and 8 have ~ the same MSE
# I prefer 6, given its higher simplicity

colMeans(MSE)
plot(colMeans(MSE))

# similar to cv.tree results
plot(y = cvtree.prostate$dev, x = cvtree.prostate$size)
# difference due to the different random
# separation of observations in the CV.
# clearly $dev is the MSE

which.min(colMeans(MSE))

# final tree:
tree.prostate = tree(lpsa~., x)
prune.prostate = prune.tree(tree.prostate_CV,
                            best = 6)
# original tree
plot(tree.prostate) #plot the tree
text(tree.prostate,pretty=0, digits=1)

# pruned tree
plot(prune.prostate) #plot the tree
text(prune.prostate,pretty=0, digits=1)

