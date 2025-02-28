# Bagging and Random Forests ####
rm(list = ls())

# load the function to compute the miscassification rate:
source("Lab/misc.R")

library(ElemStatLearn)
data(SAheart)
summary(SAheart)
head(SAheart)

n <- nrow(SAheart)
x <- SAheart[, -ncol(SAheart)] # set of predictors
y <- SAheart[, ncol(SAheart)] # response variable

heart <- data.frame(chd = as.factor(y), x)
# if we are facing a classification task, the response has to be a factor

set.seed(1234)
train <- sample(1:n, ceiling(n / 2))
heart_test <- heart[-train, ]

## Bagging ####
# install.packages('randomForest')
library(randomForest)
?randomForest
# subset
# bagging and RF have the same function: randomForest
# mtry = Number of variables randomly sampled as candidates at each split.
# by setting  mtry=ncol(x), we use all variables: Bagging.
# ntree = 500 by default
set.seed(123)
bag.heart <- randomForest(chd ~ .,
    data = heart,
    subset = train,
    mtry = ncol(x),
    importance = TRUE
)
# importance -> Should importance of predictors be assessed?
bag.heart

# does the seed matter above?
# yes, we boostrap the data.

# out of bag error rate estimate: 34.63
# Confusion matrix:
# columns = estimates
# rows = original class
# class.error = error rate for each (REAL) class
# the model is not equally accurate among the 2 classes.
# more accurate for class 0, which has more data
table(heart$chd[train])
#  0   1
# 156  75

importance(bag.heart)
library(ggplot2)
# divide plot screen in 2 rows
par(mfrow = c(2, 1))

varImpPlot(bag.heart)
varImpPlot(rf.heart)

# 2 indexes: different (similar) results
# tobacco and LDL (cholesterol): main 2 factors in coronary heart disease

yhat.bag <- predict(bag.heart,
    newdata = heart[-train, ]
)
table(yhat.bag, y[-train])

# error:
misc(yhat.bag, y[-train])
# 0.3290043
# compared to the OOB estimate of  the error rate:
bag.heart
# OOB: 33.33%

table(y[-train])
# only marginally better than assuming all observations are 0:
mean(y[-train] != 0)
# 0.3679654

# PRUNED TREE: 0.3160173 (1.1 - Trees.R)

# increasing ntree from 500 to 10,000 does NOT improve results (stable for large B)
set.seed(1234)
bag.heart_104 <- randomForest(chd ~ .,
    data = heart,
    subset = train,
    mtry = ncol(x),
    ntree = 10^4,
    importance = TRUE
)
bag.heart_104
# OOB estimate of  error rate: 33.77%
yhat.bag_104 <- predict(bag.heart_104, newdata = heart[-train, ])
misc(yhat.bag_104, y[-train])
# 0.3376623
# NOTE: OOB prediction (33.77) very similar compared to test-set prediction (0.3377)
bag.heart_104

## Random Forests ####
# Same function as before, but this time "mtry" is left unspecified
# the default values are different for
# classification (sqrt(p))
# and regression (p/3).
# randomForest will select a subset of variables at each branch.
set.seed(1234)
rf.heart <- randomForest(chd ~ .,
    data = heart,
    subset = train,
    importance = TRUE
)
rf.heart
# OOB estimate of  error rate: 33.33%

yhat.rf <- predict(rf.heart, newdata = heart[-train, ])
misc(yhat.rf, y[-train])
# 0.3246753
# similar to bagging, given the same number of trees (0.329)

varImpPlot(rf.heart)
# again Idl and tobacco are main 2 variables

# again, we increase the number of trees
# again, stable results
rf.heart_104 <- randomForest(chd ~ .,
    data = heart,
    subset = train,
    ntree = 10^4,
    importance = TRUE
)
rf.heart_104
# OOB estimate of  error rate: 33.77%
yhat.rf_104 <- predict(rf.heart_104, newdata = heart[-train, ])
misc(yhat.rf_104, y[-train])
# 0.3290043
# same error as above;

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### Exercise: Colon Dataset ####
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
rm(list = ls())
# load the function to compute the miscassification rate:
source("Lab/misc.R")

load("data/colon_data.RData")
data(colon)
dim(train)
dim(test)
# 2,000 variables (I think genes)
# MANY more predictors than before (9 above)

library(randomForest)
## Bagging ####
set.seed(1234)
bag.colon <- randomForest(as.factor(y) ~ .,
    data = train,
    mtry = ncol(train) - 1,
    importance = TRUE
)
bag.colon
# OOB error: 33.33

### Variable Importance ####
imp_bag <- importance(bag.colon)
head(imp_bag)
# sort by column 4: MeanDecreaseGini
sort_bag <- imp_bag[order(imp_bag[, 4], decreasing = T), ]
# to display the variables sorted by MeanDecreaseGini
head(sort_bag)

varImpPlot(bag.colon)
# 1 variable (gene?) dominates the bagging trees (X1671)
# remember that there are 2k variables:
varImpPlot(bag.colon, n.var = 2000)
varImpPlot(bag.colon, n.var = 100)
# few variables dominate the model

yhat.bag <- predict(bag.colon, newdata = test)
table(yhat.bag, test$y)

misc(yhat.bag, test$y)
# 0.2

table(test$y)
# most cases are 1, if we predict all 1's we get a higher error rate:
mean(test$y != 1)
# 0.35

## Random Forests ####
set.seed(1234)
rf.colon <- randomForest(as.factor(y) ~ .,
    data = train,
    importance = TRUE
)
rf.colon
# OOB error: 30.95
yhat.rf <- predict(rf.colon, newdata = test)
table(yhat.rf, test$y)

misc(yhat.rf, test$y)
# 0.35

# KEEP in mind the noise:
# the test set has 20 observations only!
# our estimate of the misclassification rate is
# unbiased, but also noisy!

### Variable Importance ####
imp_rf <- importance(rf.colon)
sort_rf <- imp_rf[order(imp_rf[, 4], decreasing = T), ]
# to display the variables sorted by MeanDecreaseGini
sort_rf[1:10, ]

varImpPlot(rf.colon)
# 1671 has lower importance than before:
# by randomly selecting variables at each split,
# we give the possibility to other variables as well.
varImpPlot(rf.colon, n.var = 2000)
varImpPlot(rf.colon, n.var = 100)
# look at the absolute values

# in bagging:
varImpPlot(bag.colon, n.var = 100)

# importance is more spread across variables than before.
