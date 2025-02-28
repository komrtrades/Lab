# LDA lab ####

misc<-function(yhat,y){
  a<-table(yhat,y)
  1-sum(diag(a))/sum(a)
}

## Univariate Case (p=1)####

#1. Derive the Bayes decision boundary
#The boundary is indeed at the midpoint of the mean values of the
# two populations -> x=0.

mu_1<--1.25
mu_2<-1.25
sigma<-1

n<-10
set.seed(ceiling(runif(1,0,10000)))
x1<-rnorm(n,mean=mu_1,sd = sigma)
x2<-rnorm(n,mean=mu_2,sd = sigma)

plot(density(x1),xlim=c(-5,5),ylim=c(0,0.8),main="Estimated kernel densities")
lines(density(x2),col="red")
(mean(x1)+mean(x2))/2
legend("topleft",legend=c("x1","x2"),col=c("black","red"),lty=1)


#2. Draw sample of 20 units each ####
n1=n2=20
set.seed(1234)
s1<-rnorm(n1,mu_1,sigma)
s2<-rnorm(n2,mu_2,sigma)

s.data<-data.frame(x=c(s1,s2),
      Population=factor(rep(1:2,c(n1,n2))))

# estimate the parameters from the samples
hat.pi1<-n1/(n1+n2)
hat.pi2<-n2/(n1+n2)
hat.mu1<-mean(s1)
hat.mu2<-mean(s2)
hat.sigma<-(sum((s1-hat.mu1)^2)+
              sum((s2-hat.mu2)^2))/(n1+n2-2)
decision.boundary.LDA<-(hat.mu1+hat.mu2)/2
decision.boundary.LDA

x<-c(s1,s2)

hat.delta1<-c(s1,s2)*hat.mu1/hat.sigma-
  hat.mu1^2/(2*hat.sigma)+log(hat.pi1)

hat.delta2<-c(s1,s2)*hat.mu2/hat.sigma-
  hat.mu2^2/(2*hat.sigma)+log(hat.pi2)

head(hat.delta1)
head(hat.delta2)

tail(hat.delta1)
tail(hat.delta2)

# compute the class label:
hat.y<-ifelse(hat.delta1>hat.delta2,1,2)
table(hat.y,s.data$Population)
table(s.data$Population,hat.y)

misc(hat.y,s.data$Population)

## Bayes classifier
bayes.labs<-ifelse(s.data$x<0,1,2)
table(bayes.labs,s.data$Population)
table(s.data$Population,bayes.labs)

misc(bayes.labs,s.data$Population)

# Plot the data
library(ggplot2)
ggplot(s.data,aes(x=x,color=Population,
                  fill=Population))+
  geom_histogram(aes(y=..density..),
                 alpha=0.5,position="identity")+
  geom_vline(aes(xintercept=decision.boundary.LDA),
        color="black",linetype="dashed",size=1)+
  geom_vline(aes(xintercept=0),color="black",size=1)+
  theme_bw()+theme(legend.position = "none")

## 3. Draw samples of 250 units each ####
n1=n2=250
set.seed(1234)
s1=rnorm(n1,mu_1,sigma)
s2=rnorm(n2,mu_2,sigma)

s.data<-data.frame(x=c(s1,s2),
Population=factor(rep(1:2,c(n1,n2))))

hat.pi1<-n1/(n1+n2)
hat.pi2<-n2/(n1+n2)
hat.mu1<-mean(s1)
hat.mu2<-mean(s2)
hat.sigma<-(sum((s1-hat.mu1)^2)+
              sum((s2-hat.mu2)^2))/(n1+n2-2)
decision.boundary.LDA2<-(hat.mu1+hat.mu2)/2
decision.boundary.LDA2

hat.delta1<-c(s1,s2)*hat.mu1/hat.sigma-
  hat.mu1^2/(2*hat.sigma)+log(hat.pi1)

hat.delta2<-c(s1,s2)*hat.mu2/hat.sigma-
  hat.mu2^2/(2*hat.sigma)+log(hat.pi2)


# Let's compute the class label:
hat.y<-ifelse(hat.delta1>hat.delta2,1,2)
table(hat.y,s.data$Population)

misc(hat.y,s.data$Population)

bayes.labs<-ifelse(s.data$x<0,1,2)
table(bayes.labs,s.data$Population)

misc(bayes.labs,s.data$Population)


# SAheart dataset ####
## 1. Perform LDA on the whole dataset ####
#install.packages("bestglm")
library(bestglm)
data("SAheart")
summary(SAheart)

library(MASS) # contains LDA
out.lda<-lda(chd~.,data=SAheart)
out.lda
out.pred<-predict(out.lda)
head(out.pred$class) # predicted labels
head(out.pred$posterior) # posterior probabilities
head(out.pred$x)# y=a^TX -> projected points along
                # the first discriminant direction
y.hat<-out.pred$class # predicted labels for each units
misc(y.hat,SAheart$chd) # training error rate

## 2. Training+Validation set
n<-nrow(SAheart)
set.seed(17)
index<-sample(1:n,ceiling(n/2),replace=F) 
# the numbers I draw end up in the training set
# the "remaining" ones end up in the validation set.
head(index)
out.lda.tr<-lda(chd~.,data=SAheart[index,])
y.hat.val<-predict(out.lda.tr,newdata=SAheart[-index,])$class

# Validation set error -> test error estimate:
misc(y.hat.val,SAheart$chd[-index])
table(SAheart$chd[-index],y.hat.val)


## 3. 5-fold CV ####
yy.hat<-rep(NA,n) # empty vector that will contain the
                  #  predicted label for each unit
err.cv<-NULL # empty vector that will include the misclassif
            # error rate for each fold
K<-5 # no. of folds
set.seed(17)
folds<-sample(1:K,n,replace=TRUE)

for(i in 1:K){
  x.val<-SAheart[folds==i,]
  x.train<-SAheart[folds!=i,]
  y.val<-SAheart$chd[folds==i]
  
  out.lda.cv<-lda(chd~.,data=x.train)
  y.hat<-predict(out.lda.cv,newdata=x.val)$class
  yy.hat[folds==i]<-y.hat
  err.cv[i]<-misc(y.hat,y.val)
}

# CV-error -> test error estimate
mean(err.cv)
table(yy.hat,SAheart$chd)



########## LDA su 3 gruppi ########
rm(list=ls())
# data extraction
iris.ve <- as.matrix(iris[iris$Species=="versicolor",-5])
iris.vi <- as.matrix(iris[iris$Species=="virginica",-5])
iris.se <- as.matrix(iris[iris$Species=="setosa",-5])

# sample statistics
n.1 <- nrow(iris.ve)
n.2 <- nrow(iris.vi)
n.3 <- nrow(iris.se)
S.1 <- var(iris.ve)
S.2 <- var(iris.vi)
S.3 <- var(iris.se)
m.1 <- colMeans(iris.ve) 
m.2 <- colMeans(iris.vi)
m.3 <- colMeans(iris.se)

x <- rbind(iris.ve, iris.vi, iris.se)

library(MASS)

# confusion matrix
classif.true <- c(rep("versicolor",n.1),rep("virginica",n.2), rep("setosa", n.3))

# 3 group analysis
out.lda <- lda(x = x, grouping = classif.true)

# unit allocation (Bayes rule)
classif.pred <- predict(object = out.lda, new.data = x)$class

# confusion matrix
table(classif.true,classif.pred)

out.lda$scaling
