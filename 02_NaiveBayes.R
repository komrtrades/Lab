# Naive Bayes Classifier ####

## Simulated example ##
library(mvtnorm) # to generate data from multivariate gaussians
library(klaR) # to run NaiveBayes function

mu0<-c(1,2)
mu1<-c(2,1.5)
sigma0<-matrix(c(2,0.5,0.5,2),2,2)
sigma1<-matrix(c(1.5,0,0,1.5),2,2)

set.seed(12345)
x0<-rmvnorm(50,mean=mu0,sigma=sigma0)
x1<-rmvnorm(50,mean=mu1,sigma=sigma1)

x<-rbind(x0,x1)
y<-rep(c(0,1),each=50)

x<-as.data.frame(x)

par(mfrow=c(1,1))
plot(x,col=ifelse(y==1,"cornflowerblue","coral"),lwd=1.5,
     xlab="X1",ylab="X2",main="Simulated example")

## Grid of points ####
x1.new<-seq(0,to=4,length.out=100)
x2.new<-seq(0,to=5,length.out=90)
x.new<-expand.grid(x=x1.new,y=x2.new)

n<-nrow(x)
K<-5
set.seed(1234)
folds<-sample(1:K,n,replace=T)
table(folds)

err.cv.g<-err.cv.k<-NULL # vectors of CV errors
yyhat.g<-yyhat.k<-rep(NA,n) # vector of predicted labels


## Cross-Validation ####
for (i in 1:K){
  x.val<-x[folds==i,]
  x.train<-x[folds!=i,]
  y.val<-y[folds==i]
  y.train<-y[folds!=i]
  
  # Naive Bayes with kernel density 
  out.nb.k<-NaiveBayes(x=x.train,grouping=as.factor(y.train),
                       usekernel=TRUE)
  y.hat.k<-predict(out.nb.k,newdata=x.val)$class
  yyhat.k[folds==i]<-y.hat.k
  err.cv.k[i]<-mean(y.hat.k!=y.val)
  
  # Naive Bayes with Gaussian density
  out.nb.g<-NaiveBayes(x=x.train,grouping=as.factor(y.train),
                       usekernel=FALSE)
  y.hat.g<-predict(out.nb.g,newdata=x.val)$class
  yyhat.g[folds==i]<-y.hat.g
  err.cv.g[i]<-mean(y.hat.g!=y.val)
}

# Confusion matrix for NB with Gaussian univariate density
yyhat.g<-yyhat.g-1
table(yyhat.g,y)
mean(yyhat.g!=y)
mean(err.cv.g)

# Confusion matrix for NB with kernel univariate density
yyhat.k<-yyhat.k-1
table(yyhat.k,y)
mean(yyhat.k!=y)
mean(err.cv.k)

# Given that the CV error is smaller with Gaussian NB, this 
# choice of density estimation is preferred.

## Bayes classifier on simulated data

## Observed density values for simulated x
## in the two sub-populations
dx.x0<-dmvnorm(x,mean=mu0,sigma=sigma0)
dx.x1<-dmvnorm(x,mean=mu1,sigma=sigma1)

# Posterior probability P(Y=1|X)
post.x<-(0.5*dx.x1)/(0.5*dx.x0+0.5*dx.x1)

# Label allocation
y.hat.x<-post.x>0.5

# Error rate
mean(y.hat.x!=y)

###########################################################

# Naive Bayes Classifier ####

## SAheart dataset ##

# Load the dataset
library(bestglm)
data('SAheart')

summary(SAheart)

# Load the klaR library for the Naive Bayes ####
library(klaR)
n<-nrow(SAheart)
x<-SAheart[,-c(5,10)] # remove categorical data and response, since you cannot compute the gaussian on binary or qualitative data
y<-SAheart[,10] # storing the response

K=5
set.seed(17)
folds<-sample(1:K,n,replace=T) #manually creating folds.
table(folds)
# knn function does resampling by itself (without reintroduction, as it should be)


err.cv.g<-err.cv.k<-NULL # vectors of CV errors
yyhat.g<-yyhat.k<-rep(NA,n) # vector of predicted labels

## Cross-Validation ####
for (i in 1:K){
  x.val<-x[folds==i,]
  x.train<-x[folds!=i,]
  y.val<-y[folds==i]
  y.train<-y[folds!=i]
  
  # Naive Bayes with kernel density 
  out.nb.k<-NaiveBayes(x=x.train,grouping=as.factor(y.train), # remember to transform qualitative variables into factors
                       usekernel=TRUE)
  y.hat.k<-predict(out.nb.k,newdata=x.val)$class
  yyhat.k[folds==i]<-y.hat.k
  err.cv.k<-mean(y.hat.k!=y.val)
  
  # Naive Bayes with Gaussian density
  out.nb.g<-NaiveBayes(x=x.train,grouping=as.factor(y.train),
                       usekernel=FALSE)
  y.hat.g<-predict(out.nb.g,newdata=x.val)$class
  yyhat.g[folds==i]<-y.hat.g
  err.cv.g<-mean(y.hat.g!=y.val)
}

# Confusion matrix for NB with Gaussian univariate density
table(yyhat.g,y)
mean(err.cv.g)

# Confusion matrix for NB with kernel univariate density
table(yyhat.k,y)
mean(err.cv.k)

# Given that the CV error is smaller with Gaussian NB, this 
# choice of density estimation is preferred.