rm(list = ls())
###################################################
# load data
###################################################
library(ElemStatLearn)
data("SAheart")

# load("data/SAheart.RData")

n <- nrow(SAheart)
x <- SAheart[, -ncol(SAheart)] # predictors
head(x)
y <- SAheart[, ncol(SAheart)] # response
table(y)
# binary classification

# SVM also works for regression.
# To do classification, we need to re-code y as factor.
heart <- data.frame(chd = as.factor(y), x)

# Training + Validation set
set.seed(1234)
train <- sample(1:n, ceiling(n / 2))
heart_valid <- heart[-train, ]

source("Lab/misc.R") # loading the misclassification function

###################################################
## Support Vector Classifier (=linear kernel) ####
###################################################
# install.packages('e1071')
library(e1071)
?svm
# Tuning parameters = ?
# 1) cost = penalty cost (default = 1)
# ‘C’-constant of the regularization term
# 2) type of kernel:
# linear (SV classifier)
# polynomial (also specify degree - 3 by default)
# radial (default)
# sigmoid
# 3.1) for polynomial -> degree
# 3.2) for radial -> gamma
set.seed(1234)
svmfit <- svm(chd ~ .,
     data = heart,
     kernel = "linear",
     cost = 10, subset = train
)
svmfit
# cost = cost of constraints violation (default: 1)
# it is the ‘C’-constant of the regularization term in the Lagrange formulation.
# cost -> same idea, but NOT THE SAME INTERPRETATION AS IN THE SLIDES!
# HERE NOT at most C misclassified training observations

# 60 misclassifications in training data
table(predict(svmfit), heart$chd[train])

svmfit$index # returns the row identifier of the support vectors

summary(svmfit)
# Number of Support Vectors:  135

# we decrease the cost (penalty)
set.seed(1234)
svmfit2 <- svm(chd ~ .,
     data = heart,
     kernel = "linear",
     cost = 0.1, subset = train
)
summary(svmfit2)

# 50 misclassifications in training data
# LOWER C -> fewer misclassifications in the training data
table(predict(svmfit2), heart$chd[train])

###################################################
# linear kernel - choose COST via CV
###################################################
# varying cost: ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
?tune
# Parameter Tuning of Functions Using Grid Search
set.seed(1234)
tune.out <- tune(
     METHOD = svm,
     chd ~ .,
     data = heart[train, ],
     kernel = "linear",
     ranges = list(cost = c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10))
)
summary(tune.out)
# 10-fold CV
# best parameters: cost 0.1

tune.out$best.model
# cost of 0.1

plot(
     y = tune.out$performances$error,
     x = tune.out$performances$cost
)

# WE CAN DO THE SAME, coding the CV on our own:
train_x <- heart[train, ]
k <- 10 # 10-folds cross-validation
temp <- c(rep(1:9, 23), rep(10, 24)) # make sure that folds are balanced

cost <- c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10)
err.cv <- matrix(NA, k, length(cost))
set.seed(1234)
for (i in 1:k) {
     for (j in 1:length(cost)) {
          # fit model on all data, except index!=i
          svm_CV <- svm(chd ~ .,
               data = train_x[temp != i, ],
               kernel = "linear",
               cost = cost[j], subset = train
          )

          # predict index==i
          yhat <- predict(svm_CV,
               newdata = train_x[temp == i, ]
          )

          # store CV error on index!=i
          err.cv[i, j] <- misc(yhat, train_x$chd[temp == i])
     }
}
err.cv
# we average the error across the K errors:
colMeans(err.cv)

plot(
     y = colMeans(err.cv),
     x = cost
)

cost[which.min(colMeans(err.cv))]
# difference, due to randomness
# 2 sources of random noise:
# 1) how data are split across the CV folds;
# 2) the SVM optimization (which is numerical).

# extract best model:
bestmod <- tune.out$best.model
summary(bestmod)
# cost:  0.1

###################################################
# joint optimization of ALL tuning parameters
###################################################
## Comparing different kernel ####
# in ranges -> we specify 3 kernels and 5 costs
# other parameters (d - polynomial and gamma - radial) are kept as default
set.seed(1234)
tune.all <- tune(svm, chd ~ .,
     data = heart[train, ],
     ranges = list(
          kernel = c(
               "linear",
               "polynomial",
               "radial"
          ),
          cost = c(0.01, 0.1, 1, 10, 100)
     )
)
# we could add gamma and d as well, but parameters are optimized in a naive way
# gamma=c(0.5,1,2,3,4,10),
# d=c(0.5,1,2,3,4))
# e.g., linear model is fit multiple times with all combinations of d and gamma

# best way -> optimize each kernel with its tuning parameters,
# then choose the model with the smallest CV error.

###################################################
# OPTIMIZE EACH KERNEL - CHOOSE TUNING PARAMETERS VIA CV
###################################################
### linear kernel
set.seed(1234)
tune_linear <- tune(
     METHOD = svm,
     chd ~ .,
     data = heart[train, ],
     kernel = "linear",
     ranges = list(cost = c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10))
)
# min of CV error:
min(tune_linear$performances$error)
# 0.2728261
# make sure we are not on the boundary of the grid:
tune_linear$best.model

### radial kernel
set.seed(1234)
tune_radial <- tune(svm, chd ~ .,
     data = heart[train, ], kernel = "radial",
     ranges = list(
          cost = c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10),
          gamma = c(0.5, 1, 2, 3, 4, 5, 10)
     )
)
min(tune_radial$performances$error)
# 0.3121377
# make sure we are not on the boundary of the grid:
tune_radial$best.model

tune_radial$best.parameters
#    cost gamma
# 6    1   0.5

# decrease gamma and re-run
tune_radial <- tune(svm, chd ~ .,
     data = heart[train, ], kernel = "radial",
     ranges = list(
          cost = c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10),
          gamma = seq(0, 1, by = 0.1)
     )
)
min(tune_radial$performances$error)
# 0.2987319
tune_radial$best.parameters
#    cost gamma
# 14  0.5   0.1

### polynomial kernel
set.seed(1234)
tune_poly <- tune(svm, chd ~ .,
     data = heart[train, ], kernel = "polynomial",
     ranges = list(
          cost = c(0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10),
          d = c(0.5, 1, 2, 3, 4)
     )
)
min(tune_poly$performances$error)
# 0.2733696
tune_poly$best.parameters

# linear has the smaller CV error:
min(tune_linear$performances$error)
min(tune_radial$performances$error)
min(tune_poly$performances$error)

###################################################
# ASSESS PERFORMANCE ON "BEST MODEL" (chosen via CV) on the TEST set
###################################################
yhat <- predict(tune_linear$best.model, newdata = heart[-train, ])

table(predict = yhat, truth = heart$chd[-train])
misc(yhat, heart$chd[-train])
# 0.2727273
# lower than what we previously obtained with tree-based models

table(heart_valid$chd)
# error rate assuming all observations are 0 (mode)
mean(heart_valid$chd != 0)
# 0.3679654

###################################################
### How does SVM compare with LDA? ####
###################################################
library(MASS)
out.lda <- lda(chd ~ ., heart, subset = train)
yhat.lda <- predict(out.lda, newdata = heart[-train, ])$class

table(predict = yhat.lda, truth = heart$chd[-train])
misc(yhat.lda, heart$chd[-train])
# 0.2640693

###################################################
# Visualize results in 2D (p = 2)
###################################################
# there are many predictors; here we only select the main 2 (from previous analyses)
# tobacco + ldl
data <- data.frame(
     chd = as.factor(SAheart$chd),
     tobacco = SAheart$tobacco,
     ldl = SAheart$ldl
)

plot(
     x = data$ldl,
     y = data$tobacco,
     col = data$chd,
     main = "", xlab = "", ylab = "", pch = 20, cex = 2
)
# linear works well, because the separation is indeed fairly linear

set.seed(1234)
svmfit <- svm(chd ~ tobacco + ldl,
     kernel = "linear",
     data = data
)
plot(svmfit, data)
# linear kernel: a line separates data
# colour of the points refers to the original class

set.seed(1234)
svmfit <- svm(chd ~ tobacco + ldl,
     kernel = "radial",
     data = data
)
plot(svmfit, data)

set.seed(1234)
svmfit <- svm(chd ~ tobacco + ldl,
     kernel = "polynomial",
     d = 2,
     data = data
)
plot(svmfit, data)

set.seed(1234)
svmfit <- svm(chd ~ tobacco + ldl,
     kernel = "polynomial",
     d = 3,
     data = data
)
plot(svmfit, data)

set.seed(1234)
svmfit <- svm(chd ~ tobacco + ldl,
     kernel = "polynomial",
     d = 4,
     data = data
)
plot(svmfit, data)
