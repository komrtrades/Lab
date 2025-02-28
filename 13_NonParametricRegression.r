######################################################
# load data:
######################################################
rm(list = ls())
data(mcycle, package = "MASS")
str(mcycle)
## 'data.frame': 133 obs. of  2 variables:
##  $ times: num  2.4 2.6 3.2 3.6 4 6.2 6.6 6.8 7.8 8.2 ...
##  $ accel: num  0 -1.3 -2.7 0 -2.7 -2.7 -2.7 -1.3 -2.7 -2.7 ...
plot(accel ~ times, data = mcycle) # would be a simple option.

######################################################
# apply kerner regression smoother
######################################################
# function:
?ksmooth
# key parameters:
# kernel: box or normal
# bandwidth = lambda
accel <- mcycle$accel # for unique values
times <- mcycle$times + rnorm(length(accel), -.001, .001)


# default smoother (bandwidth = 0.5)
k0 <- ksmooth(times, accel) # default is too coarse
lines(k0, col = 1)

# x.points
# points at which to evaluate the smoothed fit.
# If missing, n.points are chosen uniformly to cover range.x.

# we provide all points in "times"
k1 <- ksmooth(times, accel, x.points = times)
plot(times, accel)
lines(k1, col = 2)

# we play with the kernel bandwidth (lambda):

# increase the bandwidth (less smoothing - more global pattern):
k2 <- ksmooth(times, accel, bandwidth = 5, x.points = times)
lines(k2, col = 3)

# increase the bandwidth even more
k2 <- ksmooth(times, accel, bandwidth = 10, x.points = times)
lines(k2, col = 4)
# probably too much smooting:
# we do not follow the trend when going down (time 15-25)

# increase the bandwidth a lot = average of all observations!
k2 <- ksmooth(times, accel, bandwidth = 100, x.points = times)
lines(k2, col = 6)



# we play with the kernel choice:
# box kernel (default)
k1 <- ksmooth(times, accel,
  x.points = times, bandwidth = 5
)
plot(times, accel)
lines(k1, col = 2)

# normal kernel:
k3 <- ksmooth(times, accel,
  kernel = "normal",
  bandwidth = 5, x.points = times
)
lines(k3, col = 7)
# a normal kernel makes it smoother (more global behaviour)

# CHOOSE kernel and bandwidth with LEAVE-ONE-OUT CV:
# why not 10-fold CV?
# because we fit a local curve,
# so we do NOT want to exclude too many points
# (this would alter the local behaviour of the curve)
N <- length(times)
N
# make reasonable grid from above:
bandwidth <- seq(1, 10, by = 0.1)
kernel <- c("box", "normal")
grid <- expand.grid(bandwidth, kernel)
grid[, 2] <- as.character(grid[, 2])

MSE <- rep(0, nrow(grid))

set.seed(1234)
# loop over grid of values of span:
for (s in 1:nrow(grid)) {
  # loop over iterations
  for (i in 2:(N - 1)) {
    # exclude 1st and last values:
    # AVOID extrapolations!
    ks <- ksmooth(times[-i], accel[-i],
      kernel = grid[s, 2],
      bandwidth = grid[s, 1],
      x.points = times[i]
    )

    y_hat <- ks$y

    MSE[s] <- MSE[s] + (y_hat - accel[i])^2
  }
}
plot(
  y = MSE, x = grid[, 1], lwd = 3,
  pch = ifelse(grid[, 2] == "box", 1, 2)
)

cv <- grid[which.min(MSE), ]
cv
#    Var1   Var2
# 107  2.5 normal

k1 <- ksmooth(times, accel,
  x.points = times,
  bandwidth = as.numeric(cv[1]),
  kernel = as.character(cv[2])
)
plot(times, accel)
lines(k1, col = 2)
# reasonable fit

######################################################
# sm.regression (an other nonparametric)
######################################################
library(sm)

# h = smoothing parameter:
sm.regression(times, accel, h = 1)
# small values (overfitting)
sm.regression(times, accel, h = 0.5) # follow too much local behaviour
sm.regression(times, accel, h = 0.1)
# large values (too much smoothing)
sm.regression(times, accel, h = 3) # too far from cloud of points ~15-25
sm.regression(times, accel, h = 10)

# choose smoothing via CV:
# specify reasonable values for h in hstart (0.5) and hend (3)
?hcv
cv <- hcv(times, accel,
  display = "line",
  hstart = 0.5, hend = 3,
  ngrid = 100
)

cv
# 1.476231

# good smooting:
sm.regression(times, accel, h = cv)


######################################################
# local polynomials (loess)
######################################################
?loess
# Local Polynomial Regression Fitting
# key parameter:
# span - the parameter α which controls the degree of smoothing

# default smoothing value (0.75)
plot(times, accel, pch = 20)
lobj <- loess(accel ~ times)
lines(lobj$x, lobj$fitted, col = 5)
# too much smoothing!

# smaller span - not enough smoothing
lobj <- loess(accel ~ times, span = .1)
lines(lobj$x, lobj$fitted, col = 3)

# intermetiate value - # probably close to optimal
lobj <- loess(accel ~ times, span = 0.35)
lines(lobj$x, lobj$fitted, col = 2, lwd = 2)

# OLD function:
# LOWESS smoother which uses locally-weighted polynomial regression
lines(lowess(times, accel, f = 0.35), col = 4, lwd = 2) # same value

# CHOOSE span with LEAVE-ONE-OUT CV:
# why not 10-fold CV?
# because we fit a local curve,
# so we do NOT want to exclude too many points
# (this would alter the local behaviour of the curve)
N <- length(times)
N
# make reasonable grid from above:
grid_span <- seq(0.3, 0.75, by = 0.01)

MSE <- rep(0, length(grid_span))

set.seed(1234)
# loop over grid of values of span:
for (s in 1:length(grid_span)) {
  # loop over iterations
  # exclude 1st and last values:
  # we CANNOT do extrapolations!
  for (i in 2:(N - 1)) {
    lobj <- loess(accel[-i] ~ times[-i],
      span = grid_span[s]
    )

    y_hat <- predict(lobj, times[i])

    MSE[s] <- MSE[s] + (y_hat - accel[i])^2
  }
}
plot(y = MSE, x = grid_span, lwd = 3)

cv_span <- grid_span[which.min(MSE)]
cv_span
# [1] 0.35

lobj <- loess(accel ~ times, span = cv_span)

plot(times, accel, pch = 20)
lines(lobj$x, lobj$fitted, col = 2, lwd = 2)

# EYE-CHECK IS ALSO IMPORTANT in this case
