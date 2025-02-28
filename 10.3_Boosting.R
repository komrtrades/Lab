# Boosting ####
rm(list = ls())
# load the function to compute the miscassification rate:
source("Lab/misc.R")

## Regression Trees ####
# install.packages('gbm')
library(gbm)

library(ElemStatLearn)
data(prostate)
names(prostate)

# load("data/prostate.RData")

x <- prostate[, -ncol(prostate)]
p <- ncol(x) - 1
n <- nrow(x)

### Training+Validation sets ####
set.seed(1234)
train <- sample(1:n, ceiling(n / 2))
x.valid <- x[-train, ]

?gbm

# 3 tuning parameters: d, n_trees, and lambda
# 1) interaction.depth = d (default = 1)
# 2) n.trees = B
# 3) shrinkage = lambda (0.1)
# we can also choose the distribution:
# distribution defines the error and type of data
# adaboost, bernoulli (0,1), gaussian (squared error):
# Gaussian for regression
set.seed(1234)
boost.prostate <- gbm(lpsa ~ .,
  data = x[train, ],
  distribution = "gaussian",
  n.trees = 1000,
  shrinkage = 0.1,
  # cv.folds = 10,
  interaction.depth = 4
)
summary(boost.prostate)

# Let's plot the marginal effect of the selected
# variables on the response after integrating out
# the other variables.
# marginal change in y, depending on univariate predictors

# marginal effect = change in y,
# depending on univariate predictors
par(mfrow = c(2, 1))
plot(boost.prostate, i = "lweight")
plot(boost.prostate, i = "lcavol")

plot(boost.prostate, i = "gleason")
plot(boost.prostate, i = "svi")

# y (prostate-specific antigen) increases with log prostate weight
# and with log volume of cancer

yhat.boost <- predict(boost.prostate,
  newdata = x[-train, ],
  ntrees = 1000
)
mean((yhat.boost - x.valid$lpsa)^2)
# 0.9715786

### Comparing with Bagging and Random Forests ####

library(randomForest)
#### Bagging ####
set.seed(1234)
bag.prostate <- randomForest(lpsa ~ .,
  data = x,
  subset = train,
  mtry = p,
  importance = TRUE
)
bag.prostate
varImpPlot(bag.prostate)
yhat.bag <- predict(bag.prostate, newdata = x[-train, ])
mean((yhat.bag - x.valid$lpsa)^2)
# 0.438557
# much smaller than boosting

#### Random Forests ####
set.seed(1234)
rf.prostate <- randomForest(lpsa ~ .,
  data = x,
  subset = train,
  importance = TRUE
)
rf.prostate
varImpPlot(rf.prostate)

yhat.rf <- predict(rf.prostate, newdata = x[-train, ])
mean((yhat.rf - x.valid$lpsa)^2)
# 0.4639739
# similar to bagging

# So is Boosting rubbish?

# NO -> we did NOT optimize the tuning parameters!
# OPTIMIZE Boosting tuning parameters
# keep lambda - shrinkage to default

# we choose B (n.trees) and d (interaction.depth)

### Boosting: choice of B in cross-validation ####
train_x <- x[train, ]
k <- 10 # 10-folds cross-validation
temp <- c(rep(1:9, 5), rep(10, 4)) # make sure that folds are balanced
set.seed(1234)
index <- sample(temp, length(temp), replace = FALSE) # folds

# Keep lambda (shrinkage) fixed here (0.1)
# Possible values for B:
b <- c(50, 100, 200, 500, 1000, 1500, 2000, 5000) # no.iterations for boosting
# test for d = 1 and 4 nodes

#### With interaction.depth=4 ####
err.cv <- matrix(NA, k, length(b))
set.seed(1234)
for (i in 1:k) { # CV loop over the K folds
  # fit model on all data, except index!=i
  boost.out <- gbm(lpsa ~ .,
    data = train_x[index != i, ],
    distribution = "gaussian",
    n.trees = max(b),
    interaction.depth = 4,
    bag.fraction = 1
  )

  for (j in 1:length(b)) { # loop over the grid for B (n.trees)
    # predict index==i
    yhat <- predict(boost.out,
      newdata = train_x[index == i, ],
      n.trees = b[j]
    )

    # store CV error on index!=i
    err.cv[i, j] <- mean((yhat - train_x$lpsa[index == i])^2)
  }
}
err.cv
# we average the error across the K errors:
colMeans(err.cv)
best_b <- b[which.min(colMeans(err.cv))]

#### With interaction.depth=1 ####
# we do the same, with interaction.depth=1
err.cv2 <- matrix(NA, k, length(b))
set.seed(1234)
for (i in 1:k) { # CV loop over the K folds
  boost.out <- gbm(lpsa ~ .,
    data = train_x[index != i, ],
    distribution = "gaussian",
    n.trees = max(b),
    interaction.depth = 1,
    bag.fraction = 1
  )
  # no need to re-fit the model!
  # Fit the model once with max(b) trees,
  # select the first "b" trees when predicting

  for (j in 1:length(b)) { # loop over the grid for B (n.trees)
    yhat <- predict(boost.out,
      newdata = train_x[index == i, ],
      n.trees = b[j]
    )

    err.cv2[i, j] <- mean((yhat - train_x$lpsa[index == i])^2)
  }
}

plot(
  y = colMeans(err.cv),
  x = b,
  type = "l"
)
lines(
  y = colMeans(err.cv2),
  x = b, col = "red"
)
# both decrease on the left side.
# what can we do?
# try smaller values of b!

b <- seq(10, 100, by = 10) # no.iterations for boosting
# run again the CV above


best_b2 <- b[which.min(colMeans(err.cv2))]

min(colMeans(err.cv))
min(colMeans(err.cv2))

best_b
best_b2
# the optimum n.trees is different for interaction.depth 1 and 4.
# why?

# Both parameters model the complexity of the tree: there is a trade-off between them.
# With a smaller interaction depth (=1) a larger no. of trees is needed (optimum = 30-40)
# With a larger interaction depth (=4) a smaller no. of trees will work (optimum = 20)

# Overall, interaction.depth=4 with n.trees = 20 leads to a smaller CV error

#### we compare results (interaction.depth 1 and 4) on the test set now:
set.seed(1234)
# d = 4 and B = 20
best_b
boost.out.d4 <- gbm(lpsa ~ .,
  data = x[train, ],
  distribution = "gaussian",
  n.trees = best_b,
  interaction.depth = 4,
  bag.fraction = 1
)
yhat.d4 <- predict(boost.out.d4, newdata = x[-train, ], n.trees = best_b)
mean((yhat.d4 - x$lpsa[-train])^2)
# 0.4146668

set.seed(1234)
# d = 1 and B = 30
best_b2
boost.out.d1 <- gbm(lpsa ~ .,
  data = x[train, ],
  distribution = "gaussian",
  n.trees = best_b2,
  interaction.depth = 1,
  bag.fraction = 1
)
yhat.d1 <- predict(boost.out.d1, newdata = x[-train, ], n.trees = best_b)
mean((yhat.d1 - x$lpsa[-train])^2)
# 0.4885976

# Again, also on the test dataset,
# the model with lowest error is: interaction.depth=4 with n.trees = 20

# N.B. TUNING was essential.
# using default parameters we had a test-set MSE of 0.9715786.

# now (0.4146668) we have a smaller test-set MSE
# than RF (0.4639739) and bagging (0.438557)

###################################################
# do a joint optimization of ALL 3 tuning parameters:
###################################################
d <- 1:4

lambda <- seq(0.01, 0.1, by = 0.01)

b <- seq(50, 1000, by = 50)



# kind-of, but simplified with grid, so we only have 2 nested loops:
grid <- expand.grid(d, lambda, b)
colnames(grid) <- c("d", "lambda", "b")
dim(grid)
head(grid)

err.cv <- matrix(NA, k, nrow(grid))

set.seed(1234)
for (i in 1:k) { # CV loop over the K folds
  for (j in 1:nrow(grid)) { # loop over the grid for B (n.trees)
    # fit model on all data, except index!=i
    boost.out <- gbm(lpsa ~ .,
      data = train_x[index != i, ],
      distribution = "gaussian",
      n.trees = grid$b[j],
      interaction.depth = grid$d[j],
      shrinkage = grid$lambda[j],
      bag.fraction = 1
    )

    # predict index==i
    yhat <- predict(boost.out,
      newdata = train_x[index == i, ],
      n.trees = grid$b[j]
    )

    # store CV error on index!=i
    err.cv[i, j] <- mean((yhat - train_x$lpsa[index == i])^2)
  }
  print(i)
}
err.cv

# length?
length(colMeans(err.cv))

# limitation of a grid based on 3 parameters?
# hard to visualize results

min(colMeans(err.cv))
sel_tuning <- which.min(colMeans(err.cv))
grid[sel_tuning, ]
#     d lambda  b
# 107 3   0.07 30

# none on the boundary of the grid

# re-fit the model using CV parameter values above
# estimate accuracy (MSE) on the test set:
set.seed(1234)
boost_tuning <- gbm(lpsa ~ .,
  data = x[train, ],
  distribution = "gaussian",
  n.trees = grid$b[sel_tuning],
  interaction.depth = grid$d[sel_tuning],
  shrinkage = grid$lambda[sel_tuning],
  bag.fraction = 1
)
yhat_tuning <- predict(boost_tuning,
  newdata = x[-train, ],
  n.trees = grid$b[sel_tuning]
)
mean((yhat_tuning - x$lpsa[-train])^2)
# 0.4107489
# slightly better than before
# 0.4146668

###########################################################################
## Classification trees ####
###########################################################################
rm(list = ls())
source("Lab/misc.R")

library(ElemStatLearn)
data("SAheart")

# load("data/SAheart.RData")

n <- nrow(SAheart)
x <- SAheart[, -ncol(SAheart)]
y <- SAheart[, ncol(SAheart)]

heart <- data.frame(chd = as.factor(y), x)

set.seed(1234)
train <- sample(1:n, ceiling(n / 2))

# distribution="bernoulli"; (0,1) classification data.
# bag.fraction = the fraction of the training set observations randomly
# selected to propose the next tree in the expansion.
# This introduces randomnesses into the model fit (0.5 by default).
set.seed(1234)
boost.out <- gbm(chd ~ .,
  data = heart[train, ], distribution = "bernoulli",
  n.trees = 150, interaction.depth = 4, bag.fraction = 1
)
boost.out
# summary(boost.out)
# With a factor as response, it does not work.
class(heart$chd)
# We need a numeric response (as in SAheart)
class(SAheart$chd)
set.seed(1234)
boost.out <- gbm(chd ~ .,
  data = SAheart[train, ], distribution = "bernoulli",
  n.trees = 150, interaction.depth = 4, bag.fraction = 1
)
boost.out
summary(boost.out)
# age, tobacco and LDL main predictors

# response = coronary heart disease (0 or 1)
# scale of plots not [0-1]; likely transformed
# clear trends:
plot(boost.out, i = "age")
plot(boost.out, i = "tobacco")
plot(boost.out, i = "ldl")
# noisy trends
plot(boost.out, i = "alcohol")
plot(boost.out, i = "adiposity")
plot(boost.out, i = "obesity") # likely just noise
plot(boost.out, i = "famhist") # family history of heart disease
plot(boost.out, i = "sbp")
plot(boost.out, i = "obesity")

phat <- predict(boost.out,
  newdata = SAheart[-train, ],
  type = "response"
)
head(phat) # probability
yhat <- ifelse(phat > 0.5, 1, 0)

table(yhat, SAheart$chd[-train])
misc(yhat, SAheart$chd[-train])
# 0.2943723
# only slighly smaller than setting all y's to 0:
mean(y != 0)
# 0.3463203

# But we did not optimize the tuning paramters!

## OTPMIZE B - KEEPING interaction.depth=1 #####
# same n.trees as above
boost.out1 <- gbm(chd ~ .,
  data = SAheart[train, ], distribution = "bernoulli",
  n.trees = 150, interaction.depth = 1,
  bag.fraction = 1
)
summary(boost.out1)
phat1 <- predict(boost.out1, newdata = SAheart[-train, ], type = "response")
yhat1 <- ifelse(phat1 > 0.5, 1, 0)

table(yhat1, SAheart$chd[-train])
misc(yhat1, SAheart$chd[-train])
# 0.3030303

## Reduce no.of trees ####
# DO NOT RUN A NEW MODEL -> set n.trees in predict!

# TUNING PARAMETERS -> choose n.trees with a train-test approach:
## Compare results for different number of trees on the validation set ####
err.test <- rep(NA, 150)
for (i in 1:150) {
  # no need to run a new model!
  phat1.b <- predict(boost.out1,
    newdata = SAheart[-train, ],
    type = "response", n.trees = i
  )
  yhat1.b <- factor(ifelse(phat1.b > 0.5, 1, 0), levels = 0:1)
  err.test[i] <- misc(yhat1.b, SAheart$chd[-train])
}
dev.off() # cancel the previous setting on the plot window

plot(err.test,
  xlab = "n.trees",
  type = "l"
)
# minimal error for n.trees around 80
which(err.test == min(err.test))
# 76 78 79 80 82 83
# WARNING ->
# which.min would only provide the 1st values which equals the min
err.test[which.min(err.test)]
# 0.2900433

# Consider the AdaBoost model
adaboost.out1 <- gbm(chd ~ .,
  data = SAheart[train, ],
  distribution = "adaboost",
  n.trees = 150,
  interaction.depth = 1,
  bag.fraction = 1
)
summary(adaboost.out1)

# TUNING PARAMETERS -> choose n.trees with a train-test approach:
err.test <- rep(NA, 150)
for (i in 1:150) {
  # no need to run a new model!
  phat1.b <- predict(adaboost.out1,
    newdata = SAheart[-train, ],
    type = "response", n.trees = i
  )
  yhat1.b <- factor(ifelse(phat1.b > 0.5, 1, 0), levels = 0:1)
  err.test[i] <- misc(yhat1.b, SAheart$chd[-train])
}

plot(err.test,
  xlab = "n.trees",
  type = "l"
)
# minimal error for n.trees around 80
which(err.test == min(err.test))
err.test[which.min(err.test)]
# 0.2727273

# slightly lower than with a Bernoulli model (0.2900433)

###################################################
# do a joint optimization of ALL 4 tuning parameters:
###################################################
d <- 1:4
d
lambda <- seq(0.01, 0.1, by = 0.01)
lambda
b <- seq(10, 100, by = 10)
b
model <- c("adaboost", "bernoulli")

# build grid
grid <- expand.grid(d, lambda, b, model)
colnames(grid) <- c("d", "lambda", "b", "model")
grid <- as.data.frame(grid)
dim(grid)
head(grid)

### Boosting: choice of B in cross-validation ####
train_x <- SAheart[train, ]
dim(train_x)
# 23 in each fold, except the last which has 24
k <- 10 # 10-folds cross-validation
temp <- c(rep(1:9, 23), rep(10, 24)) # make sure that folds are balanced
length(temp)

err.cv <- matrix(NA, k, nrow(grid))


set.seed(1234)
index <- sample(temp, length(temp), replace = FALSE) # folds

set.seed(1234)
for (i in 1:k) { # CV loop over the K folds
  for (j in 1:nrow(grid)) { # loop over the grid for B (n.trees)
    # fit model on all data, except index!=i
    boost.out <- gbm(chd ~ .,
      data = train_x[index != i, ],
      distribution = as.character(grid$model[j]),
      n.trees = grid$b[j],
      interaction.depth = grid$d[j],
      shrinkage = grid$lambda[j],
      bag.fraction = 1
    )

    # predict index==i
    phat <- predict(boost.out,
      newdata = train_x[index == i, ],
      n.trees = grid$b[j]
    )

    yhat <- factor(ifelse(phat > 0.5, 1, 0), levels = 0:1)

    # store CV error on index!=i
    err.cv[i, j] <- misc(yhat, train_x$chd[index == i])
  }
  print(i)
}
err.cv

# length?
length(colMeans(err.cv))

min(colMeans(err.cv))
sel_tuning <- which.min(colMeans(err.cv))
grid[sel_tuning, ]
#    d lambda  b     model
# 719 3    0.1 80 bernoulli

# LAMBDA is on the boundary of the parameter space,
# so we could try to increase lambda a bit

# measure accuruacy on the test set:
boost.out <- gbm(chd ~ .,
  data = train_x,
  distribution = as.character(grid$model[sel_tuning]),
  n.trees = grid$b[sel_tuning],
  interaction.depth = grid$d[sel_tuning],
  shrinkage = grid$lambda[sel_tuning],
  bag.fraction = 1
)

# predict index==i
phat <- predict(boost.out,
  newdata = SAheart[-train, ],
  n.trees = grid$b[sel_tuning]
)
yhat <- factor(ifelse(phat > 0.5, 1, 0), levels = 0:1)

# store CV error on index!=i
misc(yhat, SAheart$chd[-train])
# 0.3116883

# this is higher than the error obtained above (0.2727273) testing on a smaller grid.
# why?

# 1) random noise -> min error via CV, does not guarantee min error on a different dataset (test data);
# 2) bias -> here we tuned parameters via CV, then we computed the error of optimal CV prarameter on a test dataset.
# above we tuned paramters on the test set -> this is correct, but the test set CANNOT be used to estimate the error,
# it provides an under-estimate of the error.


# Colon dataset ####
rm(list = ls())
source("data/misc.R")
load("data/colon_data.RData")

# we fit 2 boosting models, with 1 and 4 nodes
set.seed(1234)
boost.colon.1 <- gbm(y ~ .,
  data = train, distribution = "bernoulli",
  n.trees = 20, interaction.depth = 1, bag.fraction = 1
)
set.seed(1234)
boost.colon.4 <- gbm(y ~ .,
  data = train, distribution = "bernoulli",
  n.trees = 20, interaction.depth = 4, bag.fraction = 1
)
err.test <- matrix(NA, 20, 2)
set.seed(1234)
# on both models, we predict the test set based on 1, ..., 100 trees:
for (i in 1:20) {
  phat1.b <- predict(boost.colon.1,
    newdata = test,
    type = "response", n.trees = i
  )
  yhat1.b <- factor(ifelse(phat1.b > 0.5, 1, 0), levels = 0:1)
  err.test[i, 1] <- misc(yhat1.b, test$y)

  phat4.b <- predict(boost.colon.4,
    newdata = test,
    type = "response", n.trees = i
  )
  yhat4.b <- factor(ifelse(phat4.b > 0.5, 1, 0), levels = 0:1)
  err.test[i, 2] <- misc(yhat4.b, test$y)
  print(i)
}

plot(err.test[, 1], type = "n", ylim = c(0, max(err.test) + 0.1))
lines(err.test[, 1], type = "b", col = 1)
lines(err.test[, 2], type = "b", col = 2)

# what would you do with a plot like this?
# expand the grid -> the optimum is on the boundary of grid!
# does it go down or up after 20?

apply(err.test, 2, which.min)
# Generally, the best size of the trees is 3
err.test[3, ]
# 0.15

# very robust to multiple choices of n.trees.
# Why exactly same error?
table(test$y)
# due to classification error of 20 values
# most models lead to 17 correctly classified observations and 3 mis-classifications.

# again, we try the same with the adaboost model:
set.seed(1234)
boost.colon.1 <- gbm(y ~ .,
  data = train, distribution = "adaboost",
  n.trees = 20, interaction.depth = 1, bag.fraction = 1
)
set.seed(1234)
boost.colon.4 <- gbm(y ~ .,
  data = train, distribution = "adaboost",
  n.trees = 20, interaction.depth = 4, bag.fraction = 1
)
err.test <- matrix(NA, 20, 2)
set.seed(1234)
# on both models, we predict the test set based on 1, ..., 100 trees:
for (i in 1:20) {
  phat1.b <- predict(boost.colon.1,
    newdata = test,
    type = "response", n.trees = i
  )
  yhat1.b <- factor(ifelse(phat1.b > 0.5, 1, 0), levels = 0:1)
  err.test[i, 1] <- misc(yhat1.b, test$y)

  phat4.b <- predict(boost.colon.4,
    newdata = test,
    type = "response", n.trees = i
  )
  yhat4.b <- factor(ifelse(phat4.b > 0.5, 1, 0), levels = 0:1)
  err.test[i, 2] <- misc(yhat4.b, test$y)
  print(i)
}

plot(err.test[, 1], type = "n", ylim = c(0, max(err.test) + 0.1))
lines(err.test[, 1], type = "b", col = 1)
lines(err.test[, 2], type = "b", col = 2)

apply(err.test, 2, which.min)
# Generally, the best size of the trees is 3
err.test[3, ]
# 0.15

# similar results as with the bernoulli model
