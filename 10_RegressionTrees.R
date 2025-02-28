# Regression Trees #####
rm(list = ls())

# load the function to compute the miscassification rate:
source("Lab/misc.R")

library(devtools)
# devtools::install_github("cran/ElemStatLearn", force = TRUE)

# devtools::install_github("cran/mclust", force = TRUE)
# devtools::install_github("cran/cluster", force = TRUE)

# Load the data
library(ElemStatLearn)
data("prostate")
names(prostate)

# load("data/prostate.RData")
head(prostate)

# 9 variables + a test/tran id

# remove the "train" id (last column):
x <- prostate[, -ncol(prostate)]
p <- ncol(x) - 1 # no of features
n <- nrow(x) # no of units

# Split data into training and validation set (first n/2 obs, and second n/2 obs)
set.seed(1234)
train <- sample(1:n, ceiling(n / 2))
x.valid <- x[-train, ]
x.train <- x[train, ]

dim(x.valid)
dim(x.train)

## Grow the regression tree ####
# install.packages('tree')
library(tree)
?tree
head(x)
# y = lpsa
# x = the 8 variables before

tree.prostate <- tree(lpsa ~ ., x, subset = train)
summary(tree.prostate)
# Variables actually used
# 7 nodes
# residuals (0-mean)

tree.prostate
plot(tree.prostate) # plot the tree
text(tree.prostate, pretty = 0, digits = 3)
# interpret the branches < (left side) vs. >= (right side)
# main variable lcavol

# it's a deterministic algorithm: with different seed, results do NOT change.

## Predict the MSE onto the validation set ####
yhat_tree <- predict(tree.prostate, x.valid)

# MSE (hard to interpret)
mean((x.valid$lpsa - yhat_tree)^2) # MSE
# 1.087141

# visually compare predictions vs. observations
plot(x = x.valid$lpsa, yhat_tree)
# estimates correlate with obs but with noise.
# why do points follow horizontal lines?
# number of leaves = max number of distinct values for y_hat

## Cross-validation to determinig optimal pruning ####
?cv.tree
set.seed(1234)
cvtree.prostate <- cv.tree(tree.prostate,
       K = 5, FUN = prune.tree
)
cvtree.prostate
# contains the sequence of subtrees according to different
# values of the cost-complexity parameter.

plot(y = cvtree.prostate$dev, x = cvtree.prostate$size)

# Changing the seed changes the allocation of the observations into the k-folds.

### Prune tree- choose number of splits via CV ####
which.min(cvtree.prostate$dev)
best.terminal <- cvtree.prostate$size[which.min(cvtree.prostate$dev)]
# best.terminal contains the no. of terminal nodes that corresponds
# to a tree with the smallest deviance in CV.
best.terminal
# size of trees pruned (# branches)
# deviance: function to minimize
# k: the tuning parameter (lambda in the slides)
# high k: high penalty cost -> smaller tree

prune.prostate <- prune.tree(tree.prostate, best = best.terminal)
plot(prune.prostate)
text(prune.prostate, pretty = 0, digits = 3)

# DRAW the tree on the board

## Compute the test error estimate of the pruned tree ####
yhat_pruned <- predict(prune.prostate, x.valid)

# MSE to measure the overall ACCURACY of the model
mean((x.valid$lpsa - yhat_pruned)^2)
# 1.017454
# on the original tree, was it higher? marginally
mean((x.valid$lpsa - yhat_tree)^2) # MSE
# 1.087141

# visually compare predictions vs. observations
# 1 estimate per leave:
par(mfrow = c(1, 2))
plot(x = x.valid$lpsa, yhat_tree) # 7 estimates on the original tree
plot(x = x.valid$lpsa, yhat_pruned) # 4 estimates on the pruned tree

# keep constant ylim to compare estimates:
plot(x = x.valid$lpsa, yhat_tree, ylim = c(0, 4)) # 7 estimates on the original tree
plot(x = x.valid$lpsa, yhat_pruned, ylim = c(0, 4)) # 4 estimates on the pruned tree

# 2 estimates are the same -> same 2 regions as original tree
# 2 regions are obtained by aggregating 2+ previous regions.

# again compare original and pruned trees:
plot(tree.prostate) # plot the tree
text(tree.prostate, pretty = 0, digits = 3)

plot(prune.prostate)
text(prune.prostate, pretty = 0, digits = 3)

# when we aggregate 2 regions: what estimate do we expect?
# at the board -> prune the tree with 3 leaves.
# what can we say about the estimates?

# training set values:
prune.prostate$y
# leave/region:
prune.prostate$where
table(prune.prostate$where)

# Consider the UNCERTAINTY of each prediction
# via a confidence interval.

# mean of each region (as in the tree):
tapply(
       X = prune.prostate$y,
       INDEX = prune.prostate$where,
       FUN = mean
)


# boxplot of actual values inside each region:
par(mfrow = c(1, 1))
boxplot(prune.prostate$y ~ prune.prostate$where)

tapply(
       X = prune.prostate$y,
       INDEX = prune.prostate$where,
       FUN = quantile,
       probs = c(0.05, 0.95)
)
# confidence interval of level 0.90



# Classification trees ####
dev.off()
rm(list = ls())
source("Lab/misc.R")
library(ElemStatLearn)
# Load the data
data(SAheart)
summary(SAheart)

# save(SAheart, file = "SAheart.RData")
# load("data/SAheart.RData")

n <- nrow(SAheart)
x <- SAheart[, -10]
y <- SAheart[, 10]

heart <- data.frame(chd = as.factor(y), x)
head(heart)
# y = chd (coronary heart disease)
?SAheart

# the response variable has to be coded as factor
# otherwise tree() will fit a REGRESSION tree
# rather than a CLASSIFICATION tree.
class(heart$chd)
table(heart$chd)

tree_heart <- tree(chd ~ ., heart)
summary(tree_heart)

plot(tree_heart)
text(tree_heart, pretty = 0)

# when pruning the left side of the tree
# do we know what value will be for the region defined by
# "tobacco < 0.98"?

# there are 5 leaves with 0's and 1 leave with 1,
# but we cannot answer the question.
# because we don't know how many training observations each region has.
# n_1 = the number of observations in the 5 regions with 0;
# n_2 = the number of observations in the 1 region with 1;
# IF n_1 > n_2 -> our estimate will be 0
# IF n_1 < n_2 -> our estimate will be 1.

## Validation set approach ####
# as before, train and test sets
set.seed(1234)
train <- sample(1:n, ceiling(n / 2))
heart.train <- heart[train, ]
heart.test <- heart[-train, ]

# fully grown tree:
?tree # for classification, it uses the Gini index.
tree_heart <- tree(chd ~ ., heart, subset = train)
plot(tree_heart)
text(tree_heart, pretty = 0)
tree_heart

yhat_heart <- predict(tree_heart, heart.test, type = "class")

# Confusion matrix (truth in columns)
table(yhat_heart, heart.test$chd)
# misclassificatioin:
misc(yhat_heart, heart.test$chd)
# 0.3982684
# very high error!
table(heart.test$chd)
# what can we take as a baseline model comparison?
# 50:50?

# the modal class is more reasonable
# assuming all samples are 0 would lead to a smaller error!
mean(heart.test$chd != 0)
# 0.3679654

# Accuracy
1 - misc(yhat_heart, heart.test$chd)

## Prune the tree ####
set.seed(1234)
cv_heart <- cv.tree(tree_heart, FUN = prune.misclass)
cv_heart

# objective function, as a function of size (n branches)
plot(y = cv_heart$dev, x = cv_heart$size)
# objective function, as a function of penalty
plot(y = cv_heart$dev, x = cv_heart$k)

# why is minimum on left for size and on right for k?
# penalty is invertional to size!!!
# larger penalry -> smaller tree

best.size <- cv_heart$size[which.min(cv_heart$dev)]
best.size

?prune.misclass
prune.heart <- prune.misclass(tree_heart, best = best.size)
plot(prune.heart)
text(prune.heart, pretty = 0)

## Estimate the test error ####
yhat_pruned <- predict(prune.heart, heart.test, type = "class")

# Confusion matrix
table(yhat_pruned, heart.test$chd)
# Overall ACCURACY of the model, measured via the
# accuracy rate or misclassification rate (its complementary)
misc(yhat_pruned, heart.test$chd)
# 0.3160173

# better than on the full tree (0.3982684),
# and better than setting all estimates to 0 (error: 0.3679654)

1 - misc(yhat_pruned, heart.test$chd)
# 0.6839827

# predicting a class = taking the MODE of each region.
yhat_pruned <- predict(prune.heart, heart.test, type = "class")
head(yhat_pruned)
# however, this hides the uncertainty in the classification;
# if 100% of the observations in a region belong to class 1, I have high confidence in this classification;
# if 51% of the observations in a region belong to class 1, I have low confidence in this classification.
# in both cases, the estimated classification is 1.

# also, sometimes, 0.5 is not the ideal cutoff for separating observations.
# the (FP and FN) errors may have a different weights.
# we may want to detect "very low risk" patients to avoid surgery/treatment (with prob < 1%).

# Consider the UNCERTAINTY of each prediction
# via the estimated probability of belonging to each class.
# removing type="class", gives us the actual probabilities that observations belong to the 2 classes.
yhat_probs <- predict(prune.heart, heart.test)
head(yhat_probs)

# probability that y = 1:
table(yhat_probs[, 2])
# N.B.: in the tree, there are 3 leaves with class 0
# and 2 leaves with class 0.
# In fact, there are 3 leaves with estimated Pr(y=1) < 0.3,
# and 2 leave with estimated Pr(y=1) > 0.5

# 39 % of patients have an estimated prob < 15% of having the condition.
mean(yhat_probs[, 2] < 0.15)
# [1] 0.3896104

# additionally, we could compute an approximate Wald-type CI for each probability
# WARNING: NOT ACCURATE when pi close to 0 or 1
# (i.e., on the boundary of its parameter space)
# CI_(1-alpha) = pi +- q_(1-alpha/2) * sqrt(pi*(1-p)/n)
pi <- tapply(prune.heart$y,
       INDEX = prune.heart$where,
       function(x) {
              mean(x == 1)
       }
)
n <- tapply(prune.heart$y,
       INDEX = prune.heart$where,
       length
)
qnorm(0.975)

# 0.95 Wald-type CI:
cbind(
       pi = pi,
       LB = pi - qnorm(0.975) * sqrt(pi * (1 - pi) / n),
       UB = pi + qnorm(0.975) * sqrt(pi * (1 - pi) / n)
)
# not accurate for pi = 1

# what probabilities have HIGH classification uncertainty?
# what probabilities have LOW classification uncertainty?

# HIGH uncertainty -> 0.5 (equal probability for the 2 classes).
# LOW uncertainty -> 0 or 1 (high confidence about one class).

# GENERAL QUESTION:
# WHY DID WE USE A TRAIN-TEST APPROACH IF WE USE CV?

# TRAIN-TEST -> only to estimate the accuracy of the model.
# (MSE or misclassificatioin rate).

# CV within the training set -> to choose the tuning parameters.

# typically, we'd re-fit the model on the entire dataset
# again, choosing the tuning parameters via CV.
