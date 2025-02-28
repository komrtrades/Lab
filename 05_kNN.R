# Bayes classifier #

## Simulated example ##
library(mvtnorm) # to generate data from multivariate gaussians

mu0<-c(1,2)
mu1<-c(2,1.5)
sigma0<-matrix(c(2,0.5,0.5,2),2,2)
sigma1<-matrix(c(1.5,0,0,1.5),2,2)

set.seed(1234)
x0<-rmvnorm(6,mean=mu0,sigma=sigma0)
x1<-rmvnorm(6,mean=mu1,sigma=sigma1)

x<-rbind(x0,x1)
y<-rep(c(0,1),each=6)

plot(x,col=ifelse(y==1,"cornflowerblue","coral"),lwd=1.5,
     xlab="X1",ylab="X2",main="Simulated example")

## Grid of points ####
x1.new<-seq(0,to=4,length.out=100)
x2.new<-seq(0,to=5,length.out=90)
x.new<-expand.grid(x=x1.new,y=x2.new)

## Bayes decision boundary ####
# Compute the probability of observing each point according
# to the two populations

dx.new0<-dmvnorm(x.new,mean=mu0,sigma=sigma0)
dx.new1<-dmvnorm(x.new,mean=mu1,sigma=sigma1)

# Posterior probability P(Y=0|X)
post.xnew.0<-(0.5*dx.new0)/(0.5*dx.new0+0.5*dx.new1)
post.xnew.0.m<-matrix(post.xnew.0,length(x1.new),
                      length(x2.new))
dev.off()
contour(x1.new,x2.new,post.xnew.0.m,level=0.5,
        labels="",xlab="X1",ylab="X2",
        main="Bayes decision boundary")
points(x.new,pch=".",cex=1.2,
       col=ifelse(post.xnew.0>0.5,"coral","cornflowerblue"))

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
mean(y.hat.x!=y) # this is only an estimate of the true error rate.

## k-NN Classifier ##

#install.packages('class') 
library(class) # library that includes knn

### 1-NN #####
out1<-knn(train=x,cl=y,test=x.new,k=1,prob=T)
str(out1)
prob1NN<-attributes(out1)$prob # all the probs are =1
prob1NN<-ifelse(out1==0,prob1NN,1-prob1NN)
prob1NN.m<-matrix(prob1NN,length(x1.new),length(x2.new))

### 3-NN #####
out3<-knn(train=x,cl=y,test=x.new,k=3,prob=T)
prob3NN<-attributes(out3)$prob # all the probs are >=0.67
prob3NN<-ifelse(out3==0,prob3NN,1-prob3NN)

prob3NN.m<-matrix(prob3NN,length(x1.new),length(x2.new))


### 5-NN #####
out5<-knn(train=x,cl=y,test=x.new,k=5,prob=T)
prob5NN<-attributes(out5)$prob # all the probs are >=0.6
prob5NN<-ifelse(out5==0,prob5NN,1-prob5NN)

prob5NN.m<-matrix(prob5NN,length(x1.new),length(x2.new)) # as k increases, the probability can take more values.

## Let's plot the results ####
par(mfrow=c(1,3))

### Plot no.1 +++ 1-NN ####
contour(x1.new,x2.new,prob1NN.m,level=0.5, labels="",
        xlab="X1",ylab="X", main="1NN decision boundary")

# To color the allocation areas:
points(x.new,pch=".",cex=1.2,
       col=ifelse(prob1NN>0.5,"coral","cornflowerblue"))

# Add the original points
points(x,col=ifelse(y==1,"cornflowerblue","coral"))


### Plot no.2 +++ 3-NN ####
contour(x1.new,x2.new,prob3NN.m,level=0.5, labels="",
        xlab="X1",ylab="X2", main="3NN decision boundary")

# To color the allocation areas:
points(x.new,pch=".",cex=1.2,
       col=ifelse(prob3NN>0.5,"coral","cornflowerblue"))

# Add the original points
points(x,col=ifelse(y==1,"cornflowerblue","coral"))


### Plot no.3 +++ 5-NN ####
contour(x1.new,x2.new,prob5NN.m,level=0.5, labels="",
        xlab="X1",ylab="X2", main="5NN decision boundary")

# To color the allocation areas:
points(x.new,pch=".",cex=1.2,
       col=ifelse(prob5NN>0.5,"coral","cornflowerblue"))

# Add the original points
points(x,col=ifelse(y==1,"cornflowerblue","coral"))

## K-NN classifier for SAheart dataset ####
#install.packages("bestglm")
library(bestglm)
data("SAheart")
n<-nrow(SAheart)
x<-SAheart[,-c(5,10)] # remove response and categorical data
y<-SAheart[,10] # the response

library(class)
set.seed(1234)
index<-sample(1:n,ceiling(n/2),replace=F)

train<-x[index,]
test<-x[-index,]
train_y<-y[index]
test_y<-y[-index]
train_std<-scale(train,T,T) #standardizes the data
ntrain<-nrow(train)

### 5-fold CV ####
K<-5
set.seed(1234)
folds<-sample(1:K,ntrain,replace=T)
k<-c(1,3,5,15,25,50) # the no. of neighbours
err.cv<-matrix(NA,K,length(k),
               dimnames = list(NULL,paste0("K=",k)))
for(i in 1:K){
  x.val<-train_std[folds==i,]
  y.val<-train_y[folds==i]
  x.train<-train_std[folds!=i,]
  y.train<-train_y[folds!=i]
  
  for (j in 1:length(k)){
    y.hat<-knn(train=x.train,test=x.val,cl=y.train,k=k[j])
    err.cv[i,j]<-mean(y.hat!=y.val)
  }
}

err.cv
colMeans(err.cv)
which.min(apply(err.cv,2,mean))


# Before estimating the test error, we need to standardize
# the data using the parameters of the training data.
mean_x<-apply(train,2,mean) # computes the mean of each column
sd_x<-apply(train,2,sd) # computes the sd of each column

# Via for-cycle
test_std<-test
for (j in 1:ncol(test)){
  test_std[,j]<-(test[,j]-mean_x[j])/sd_x[j]
}

apply(test_std,2,mean)
apply(test_std,2,sd)

# Alternatively, via matrix product
mean.m<-matrix(mean_x,nrow(test),ncol(test),
               byrow=T)
sd.m<-matrix(sd_x,nrow(test),ncol(test),
               byrow=T)
test_std2<-(test-mean.m)/sd.m


## Prediction on the standardized test set ####
y.hat<-knn(train=train_std,test=test_std,cl=train_y,k=25)

mean(y.hat!=test_y)
table(y.hat,test_y)

# knn is doing worse than naive bayes because n is small and p is large, and the true f is probably simpler.
