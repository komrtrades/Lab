## Main functions in R

## data.set.train is the reference train data-set with x (predictor) and y (target)##
## data.set.test is the reference test data-set with x (predictor) and y (target)##

## Polynomial regression with exponent i
glm.fit<-glm(y ~ poly(x, i), data = data.set.train)
## LOOCV error
glm.fit<-cv.glm(data.set.train, glm.fit)$delta[1]
## 10-fold error
cv.glm(Auto, glm.fit, K = 10)$delta[1]
## Bootstrap of exponential regression with exponent 2
boot.fn <- function(data, index)
  coef(
    lm(y ~ x + I(x),
       data = data.set.train, subset = index)
  )
boot(data.set.train, boot.fn, 1000)

## 5-fold CV
K=5
set.seed(15)
folds<-sample(1:K,n,replace=T)
table(folds)
##
for (i in 1:K){
data.set.train<-data.set[folds!=i,]
data.set.test<-data.set[folds==i,]
}

## LDA fit
library(MASS)
lda.fit<-lda(y~.,data=data.set.train)
## LDA labels
predict(lda.fit,newdata=data.set.test)$class
## misclassification error rate
misc<-function(yhat,y){
  a<-table(yhat,y)
  1-sum(diag(a))/sum(a)
}

## Logistic regression fit
out.log.cv<-glm(chd~.,data=data.set.train,family="binomial")
## Logistic regression predicted probabilities
p.hat<-predict(out.log.cv,newdata=data.set.test,type="response")
## Logistic regression predicted labels
y.hat<-ifelse(p.hat>=0.5,1,0)
## Compare two nested models
anova(model.big,model.small)

## Onwards, data.set.train$x represents the matrix of predictors.
## In case p>1, it should be identified as the matrix
## with the p predictors as columns.

## Naive Bayes - Kernel
out.nb.k<-NaiveBayes(x=data.set.train$x,grouping=as.factor(data.set.train$y),
                     usekernel=TRUE)
## Naive Bayes - Gaussian
out.nb.g<-NaiveBayes(x=data.set.train$x,grouping=as.factor(data.set.train$y),
                     usekernel=FALSE)

## Validation set 25%
set.seed(15)
ind_val<-sample(1:n,ceiling(n/4),replace=F)
ind_tr<-setdiff(1:n,ind_val)

data.set.train<-data.set[ind_tr,]
data.set.train<-data.set.test[ind_val,]

## Lift curve
out.log.train.val<-glm(chd~.,data=data.set.train,family="binomial")
pred_val<-predict.glm(out.log.train.val, newdata = data.set.test,type = 'response')
library(ROCR)
pred<-prediction(pred_val, labels = data.set.test$y)
perf <- performance(pred,"lift","rpp")
plot(perf,main="Lift curve")

## 25-NN classifier               
out1<-knn(train=data.set.train$x,cl=data.set.train$y,test=data.set.test$x,k=1,prob=T)  

##PCA
pr.out <-prcomp(data.set.train, scale = TRUE)
pr.out$x ## projections (PC scores)
pr.out$rotation # PC coefficients
pr.out$sdev # PC standard deviations

##PC scores via SVD
x.std<-scale(USArrests,T,T) # standardize the data
svd_x<-svd(x.std)
pc2<-svd_x$u[,1:2]%*%diag(svd_x$d)[1:2,1:2]

##PCR
library(pls)
pcr.fit <- pcr(y ~ ., data = data.set.train,
               scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")
pcr.pred <- predict(pcr.fit, data.set.test$x, ncomp = 5)
## PLS
library(pls)
pls.fit <- plsr(y ~ ., data = data.set.train, 
                scale = TRUE, validation = "CV")
validationplot(pls.fit, val.type = "MSEP")
pls.pred <- predict(pls.fit, data.set.test$x, ncomp = 1)
##
