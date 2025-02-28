library(ISLR2)

dim(Hitters)

summary(Hitters)

Hitters_no_na<-na.omit(Hitters)

dim(Hitters_no_na)

## Validation set approach

set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Hitters_no_na),
                replace = TRUE)
test <- (!train)

###
x <- model.matrix(Salary ~ ., Hitters_no_na)[, -1]
y <- Hitters_no_na$Salary

## PCR and PLS Regression

### Principal Components Regression

###
library(pls)
set.seed(2)
pcr.fit <- pcr(Salary ~ ., data = Hitters_no_na, scale = TRUE,
               validation = "CV")
###
summary(pcr.fit)
###
validationplot(pcr.fit, val.type = "MSEP")
###
set.seed(1)
pcr.fit <- pcr(Salary ~ ., data = Hitters_no_na, subset = train,
               scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")
###
pcr.pred <- predict(pcr.fit, x[which(test), ], ncomp = 5)
y.test<-y[which(test==TRUE)]
mean((pcr.pred - y.test)^2)
###
pcr.fit <- pcr(y ~ x, scale = TRUE, ncomp = 5)
summary(pcr.fit)

### Partial Least Squares

###
set.seed(1)
pls.fit <- plsr(Salary ~ ., data = Hitters_no_na, subset = train, scale = TRUE, validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")
###
pls.pred <- predict(pls.fit, x[which(test), ], ncomp = 3)
mean((pls.pred - y.test)^2)
###
pls.fit <- plsr(Salary ~ ., data = Hitters_no_na, scale = TRUE,
                ncomp = 3)
summary(pls.fit)
###