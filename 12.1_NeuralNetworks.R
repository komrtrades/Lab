rm(list = ls())
## regression - neural networks

## dataset Boston
library(nnet)
library(MASS)
help(Boston)
# Housing Values in Suburbs of Boston
data <- Boston
dim(data)
# n = 506 observations
# 14 variables

# aim: predict the median price of a house (Y = medv)
# from the remaining 13 predictors.

summary(data)
help(Boston)
help(nnet)

## training e test set
n.tr <- round(nrow(data) * 0.75)
set.seed(123456)
unita.training <- sample(1:nrow(data), size = n.tr)
training.set <- data[unita.training, ]
test.set <- data[-unita.training, ]
dim(training.set) # 380 obs
dim(test.set) # 126 obs

## Standardize the data (zero-mean and unitary variance)
## 1 Preliminary analyses:
# To use the ```nnet``` software effectively,
# it is essential to scale the problem. (Venables & Ripley, 2002)
# ?scale
data.scaled <- as.data.frame(scale(data,
  center = TRUE,
  scale = TRUE
))
colMeans(data.scaled) == 0
# why not?
# numerical approximation!!!
colMeans(data.scaled) # ~10^-16
# mean is 0
cov(data.scaled)
# variance is 1

training.set.scaled <- data.scaled[unita.training, ]
test.set.scaled <- data.scaled[-unita.training, ]

# fit a NN
# check key parameters on slides
help(nnet) # single-hidden-layer neural network
# y is continuous -> regression NN
head(training.set.scaled$medv)
# default lambda?
# linout  = T or F?
# M  = ?
# maxit why increasing it?
# it's more likely the method will converge
nnh <- nnet(
  formula = medv ~ .,
  data = training.set.scaled,
  trace = FALSE,
  linout = TRUE,
  size = 5, # M
  maxit = 10000
)
nnh
# a 13-5-1 network with 76 weights
# p = 13 predictors; M = 5 hidden layers; K = 1 outputs
# (P+1) * M + (M+1) * K
14 * 5 + 6
summary(nnh)
# what are these numbers?
# all weights from each predictor to each hidden neuron
# and from each hidden neuron to the output

# i1->h1 = alpha_11 = from input 1 to hidden layer 1
# i2->h1 = alpha_12 = from input 1 to hidden layer 2
# ...
# h5->o = beta_5 = from hidden layer 5 to output

# compare the computational cost of varying M
# parameters = (P+1) * M + (M+1) * K = M * (P + 1 + K) + K -> linear in M

# system.time will not be effective here (cost is too small):
# also system.time analyses a single iteration -> is affected by noise of individual runs
system.time({
  nnet(
    formula = medv ~ ., data = training.set.scaled, linout = TRUE,
    trace = FALSE, maxit = 10000, size = 2
  )
})

# microbenchmark runs a command multiple times (defined by "times"):
library(microbenchmark)
if (TRUE) {
  microbenchmark(
    nnet(
      formula = medv ~ ., data = training.set.scaled, linout = TRUE,
      trace = FALSE, maxit = 10000, size = 2
    ),
    nnet(
      formula = medv ~ ., data = training.set.scaled, linout = TRUE,
      trace = FALSE, maxit = 10000, size = 5
    ),
    nnet(
      formula = medv ~ ., data = training.set.scaled, linout = TRUE,
      trace = FALSE, maxit = 10000, size = 10
    ),
    nnet(
      formula = medv ~ ., data = training.set.scaled, linout = TRUE,
      trace = FALSE, maxit = 10000, size = 20
    ),
    times = 10
  )
}
# mean
17.05566 # M = 2
80.08952 # M = 5
300.57220 # M = 10
2094.94823 # M = 20

# 2 -> 5: ~5 times slower
80.08952 / 17.05566
# 2 -> 10: ~18 times slower
300.57220 / 17.05566
# 2 -> 20: ~123 times slower
2094.94823 / 17.05566

# why not 2.5, 5 and 10 times (increase in M)?
# number of parameters N = (p+1) * M + (M + 1) * K = M * (p + 1 + K) + K
# in our case K = 1, so  N = M * (p + 2) + 1
# N is linear wrt to M.
# increasing M by 10, increases the parameters by ~10 times;
# BUT minimizing a more complex function can require more iterations
# therefore the overall cost can increase by a factor >>> than the increase in M

# we optimize the tuning parameters...which are?
# how do we optimize them?

## cross validataon: choosing lambda (penalty for the size of theta) and M (number of hidden layers)
library(ipred)
?errorest
Ms <- c(2, 4, 6, 8)
lambdas <- c(0.001, 0.005, 0.01)
# we create a grid of values for Ms and lambdas:
# one has each, the other does not:
size <- rep(Ms, each = length(lambdas))
size
lambda <- rep(lambdas, length(Ms))
lambda
sl <- cbind(size, lambda)
sl
# alternatively, we'd use:
expand.grid(Ms, lambdas)

# output error:
out <- rep(NA, nrow(sl))

set.seed(123456)
# for each pair (M*, lambda*)
for (i in 1:nrow(sl)) {
  set.seed(123456)
  e <- errorest(medv ~ .,
    model = nnet, estimator = "cv",
    data = training.set.scaled,
    size = sl[i, 1], decay = sl[i, 2], linout = TRUE,
    maxit = 10000, trace = FALSE
  )
  out[i] <- e$err
}

# make a 3D plot:
library(scatterplot3d)
scatterplot3d(
  x = sl[, 1],
  y = sl[, 2],
  z = out,
  type = "p",
  xlab = "size",
  ylab = "decay",
  zlab = "CV error"
)
# do you understand where the minimum is?

# interactive 3D plot:
library(rgl)
plot3d(
  x = sl[, 1],
  y = sl[, 2],
  z = out,
  type = "s",
  radius = .1,
  xlab = "size",
  ylab = "decay",
  zlab = "CV error"
)

# 3d plots are not very popular...why?
# often not very clear to explore!
# alternative way to plot 3d in 2d?
# keep track of size via colours and/or shapes!
plot(
  x = sl[, 2],
  y = out,
  type = "p",
  xlab = "decay",
  ylab = "CV error",
  col = sl[, 1], # colours
  pch = sl[, 1], # shape
  lwd = 2
) # line width
legend(
  x = 0.007, y = 0.5,
  legend = unique(sl[, 1]),
  pch = unique(sl[, 1]),
  col = unique(sl[, 1]),
  lwd = 2,
  lty = 0
) # no line
# size 6 in general gives better performance,
# given each decay point

# much easier in ggplot:
library(ggplot2)
DF <- data.frame(
  size = as.factor(sl[, 1]),
  lambda = sl[, 2],
  CV_error = out
)
ggplot(DF, aes(x = lambda, y = CV_error)) +
  geom_point(aes(shape = size, col = size, size = 2))

out.cv <- cbind(sl, out^2)
sel_out_CV <- out.cv[which.min(out.cv[, 3]), 1:2]
sel_out_CV

lambdas
# min is on the boundary of lambda:
# we extend the boundary further in that direction
lambdas <- c(0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
Ms <- c(2, 4, 6, 8, 10, 12)
size <- rep(Ms, each = length(lambdas))
lambda <- rep(lambdas, length(Ms))
sl <- cbind(size, lambda)
out <- rep(NA, nrow(sl))
# and re-run above loop (line 145)

# nnet does NOT have a multi-start procedure!
## We study how variable results are,
# when runing the NN multiple times.
# we use a multi-start approach and
# run the algorithm n_iter times:
n_iter <- 10
set.seed(123456)
value <- rep(NA, n_iter)
best <- Inf # we keep the best model
# we initialize "best" as Inf, then
# if nnh$value < best, we update the optimal nnh model
for (h in 1:n_iter) {
  nnh <- nnet(medv ~ .,
    data = training.set.scaled,
    size = sel_out_CV[1], decay = sel_out_CV[2],
    maxit = 10000,
    linout = TRUE, trace = FALSE
  )
  value[h] <- nnh$value
  # value
  # value of fitting criterion plus weight decay term.
  print(paste("h, valore:", h, round(nnh$value,4), "convergence",
             nnh$convergence))
  if (nnh$value < best) {
    best <- nnh$value
    nn <- nnh
  } # do we need an else?
}
# a lot of noise:
plot(value)
plot(sort(value))
min(value)
27.26038
# maybe we can do better by increasing n_iter
n_iter <- 500

# we would like a plateau on the left side,
# indcating that several models (starts) give us
# a similar value of the error.
# a lack of plateau suggests that we could improve
# adding more starts.

# IDEALLY, we'd like to repeat the multi-start approach for each value of the grid (M, lambda) !
# we do that correctly below (line ~316).

summary(nn) # 61 weights

## performance evaluation on the test set:
Y.previsto.nn.ts.scaled <- predict(nn,
  newdata = test.set.scaled
)
Y.previsto.nn.ts <- Y.previsto.nn.ts.scaled * sd(data$medv) + mean(data$medv)

MSE.nn.ts <- mean((test.set$medv - Y.previsto.nn.ts)^2)
MSE.nn.ts
# 10.91842

## linear multiple regression
rl <- lm(log(medv) ~ crim + zn + indus + chas + I(nox^2) + I(rm^2) + age + log(dis) + log(rad) +
  +tax + ptratio + black + log(lstat), data = training.set)

# check residuals (similar checks are not possible on NN!)
par(mfrow = c(1, 2))
plot(rl, which = 1)
plot(rl, which = 2)
# in LM we can interpret coefficients:
# - direction on response (+ or -);
# - magnitude of effect (absolute value of Beta);
# - significance of predictors (test).
summary(rl)

Y.previsto.rl.ts <- predict(rl, test.set)

MSE.rl.ts <- mean((test.set$medv - exp(Y.previsto.rl.ts))^2)
MSE.rl.ts
# 20.31331

# compare with NN error:
MSE.nn.ts / MSE.rl.ts
# NN MSE is ~47% lower than LM MSE!

## Classification NN
rm(list = ls())
library(nnet)
 data <- read.csv2("data/Tayko_allpart.csv", header = TRUE)
dim(data)
# A company aims to select the customers which have
# the highest probability of purchasing its products
data <- data[, -c(1, 24)]
head(data)
# response variable: purch (yes/no)
table(data$purch)

# class has to be factor:
data$purch <- factor(data$purch)

## training, validataon e test set
# why do we split the data in 3 parts?
data.tr <- data[data$part == "t", 1:22]
data.vl <- data[data$part == "v", 1:22]
data.ts <- data[data$part == "s", 1:22]
# we typically do CV WITHIN the trainig set
# to tune parameters.
# here, since NN are computationally expensive,
# we use a test-set approach WITHIN the training set
# (i.e., split twice the data)
# then re-fit the model with those parameters on

head(data)
# most variables are 0,1 -> no need to normalize them.

## normalization: (data - min)/(max - min)
# most popular normalization for classification with NN
# now predictors belong to (0,1)
data.scaled <- data
data.scaled$freq <- (data$freq - min(data$freq)) / (max(data$freq) - min(data$freq))
data.scaled$last <- (data$last - min(data$last)) / (max(data$last) - min(data$last))
data.scaled$first <- (data$first - min(data$first)) / (max(data$first) - min(data$first))

## training, validataon e test set
data.scaled.training <- data.scaled[data.scaled$part == "t", 1:22]
data.scaled.validataon <- data.scaled[data.scaled$part == "v", 1:22]
data.scaled.test <- data.scaled[data.scaled$part == "s", 1:22]

# VALIDATION-SET to choose M and lambda
Ms <- c(1, 2, 3, 4, 5)
lambdas <- c(0.0001, 0.001, 0.01)
size <- rep(Ms, each = length(lambdas))
lambda <- rep(lambdas, length(Ms))
sl <- cbind(size, lambda)

expand.grid(Ms, lambdas)

out <- matrix(NA, nrow(sl), 1)
for (j in 1:nrow(sl)) {
  set.seed(123456)
  cat(" size=", sl[j, 1], "decay=", sl[j, 2], "|| ")
  best <- Inf
  # multi-start:
  for (h in 1:10) {
    nnh <- invisible(nnet(purch ~ .,
      data.scaled.training,
      trace = FALSE,
      size = sl[j, 1],
      decay = sl[j, 2],
      linout = FALSE, # no linear output function (logistic activation)
      maxit = 10000
    ))
    if (nnh$value < best) {
      best <- nnh$value
      nn <- nnh
    }
  }
  pred <- predict(nn, newdata = data.scaled.validataon)
  pred <- as.numeric(pred > 0.5) # regola di Bayes
  truth <- as.numeric(data.scaled.validataon$purch == "yes")
  er <- 1 - mean(truth == pred) # tasso di errata classificazione
  print(paste("er=", round(er, 3)))
  out[j] <- er
}

out.cv <- cbind(sl, TassoErroreValid = out)
sel_out_CV <- out.cv[which.min(out.cv[, 3]), 1:2]
sel_out_CV

# in ggplot:
library(ggplot2)
DF <- data.frame(
  size = as.factor(sl[, 1]),
  lambda = sl[, 2],
  CV_error = out
)
ggplot(DF, aes(x = lambda, y = CV_error)) +
  geom_point(aes(shape = size, col = size, size = 2))
# are we satisfied with the optimization?

# maybe add a few values for lambda between 0.001 and 0.01
lambdas <- c(0.0001, 0.001, 0.002, 0.005, 0.01, 0.015)
size <- rep(Ms, each = length(lambdas))
lambda <- rep(lambdas, length(Ms))
sl <- cbind(size, lambda)
# re-run loop (row 357)

## NOW, we re-run a multi-start with optimal size and lambda parameters:
n_iter <- 100
set.seed(123456)
best <- Inf
value <- rep(NA, n_iter)
for (h in 1:n_iter) {
  nnh <- nnet(purch ~ .,
    data = data.scaled.training,
    size = sel_out_CV[1],
    decay = sel_out_CV[2], maxit = 10000,
    trace = FALSE
  )
  value[h] <- nnh$value
  # print(paste("h, valore:", h, round(nnh$value,4), "convergence",nnh$convergence))
  if (nnh$value < best) {
    best <- nnh$value
    nn <- nnh
  }
}
plot(value)
plot(sort(value))
# IDEALLY, we'd like to repeat the multi-start approach for each value of the grid (M, lambda) !
# this time, we did!

summary(nn) # 93 weights

## performance evaluation on the test data:
# NN predict the probability of a class
p.nn.ts <- predict(nn, newdata = data.scaled.test)
head(p.nn.ts)
Y.previsto.nn.ts <- predict(nn,
  newdata = data.scaled.test,
  type = "class"
)
# with type = "class" it predicts the class:
head(Y.previsto.nn.ts) # yes (p > 0.5); no (p < 0.5)

mconf.nn.ts <- table(Y.previsto.nn.ts, data.scaled.test$purch)
mconf.nn.ts

# error rate
tasso.errore.nn.ts <- 1 - sum(diag(mconf.nn.ts)) / sum(mconf.nn.ts)
tasso.errore.nn.ts
# 0.186

# How else can we assess the classification accuracy?
# we have a prob -> we can build a ROC curve!
library(pROC)
?roc
truth <- ifelse(data.scaled.test$purch == "yes", 1, 0)
roc_score <- roc(truth, p.nn.ts) # AUC score
library(ggplot2)
ggroc(roc_score)

# what is the error rate compared to the ROC?

# a point of the ROC, obtained thresholding
# probabilities at 50%.

## we compare against a glm, with logit link
# built on training data
rlog <- glm(purch ~ ., data.tr,
  family = binomial(link = "logit")
)
summary(rlog)
# predict test data
p.rlog.ts <- predict(rlog, newdata = data.ts, type = "response")

Y.previsto.rlog.ts <- p.rlog.ts > 0.5
Y.previsto.rlog.ts <- as.integer(Y.previsto.rlog.ts)

mconf.rlog.ts <- table(Y.previsto.rlog.ts, data.ts$purch)
tasso.errore.rlog.ts <- 1 - sum(diag(mconf.rlog.ts)) / sum(mconf.rlog.ts)
tasso.errore.rlog.ts
# 0.164
# lower than NN ( 0.186 )

## compare models with 2 ROC curves
truth <- ifelse(data.ts$purch == "yes", 1, 0)
roc_glm <- roc(truth, p.rlog.ts) # AUC score
library(ggplot2)
ggroc(list(
  NN = roc_score,
  glm = roc_glm
))
# we can easily plot both curves together
