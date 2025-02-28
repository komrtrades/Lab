# Logistic Regression ####

# Loading the dataset ####
#install.packages("bestglm")
library(bestglm)
data("SAheart")

pairs(SAheart[,-ncol(SAheart)],col=SAheart$chd+1,lwd=1.5)
# Alternatively, if you want to use specific colors:
# col=ifelse(SAheart$chd==1,"coral","cornflowerblue")

# Simple logistic regression ####

## sbp predictor ####
out.log1<-glm(chd~sbp,data=SAheart,family="binomial")
summary(out.log1)

# e.g. for an individual with 120 sbp
#p.x<-exp(out.log1$coefficients[1]+out.log1$coefficients[2]*120)/
#  (1+exp(out.log1$coefficients[1]+out.log1$coefficients[2]*120))


## tobacco predictor ####
out.log2<-glm(chd~tobacco,data=SAheart,family="binomial")
summary(out.log2)

## ldl predictor ####
out.log3<-glm(chd~ldl,data=SAheart,family="binomial")
summary(out.log3)

## adiposity predictor ####
out.log4<-glm(chd~adiposity,data=SAheart,family="binomial")
summary(out.log4)

## famhist predictor ####
out.log5<-glm(chd~famhist,data=SAheart,family="binomial")
summary(out.log5)

## typea predictor ####
out.log6<-glm(chd~typea,data=SAheart,family="binomial")
summary(out.log6)

## obesity predictor ####
out.log7<-glm(chd~obesity,data=SAheart,family="binomial")
summary(out.log7)

## alcohol predictor ####
out.log8<-glm(chd~alcohol,data=SAheart,family="binomial")
summary(out.log8)

## age predictor ####
out.log9<-glm(chd~age,data=SAheart,family="binomial")
summary(out.log9)

# Multiple Logistic Regression ####
out.log<-glm(chd~.,data=SAheart,family="binomial")
summary(out.log)

# Multiple Logistic regression on a subset of variables ####
out.log.vs<-glm(chd~tobacco+ldl+famhist+typea+age,family="binomial",
                data=SAheart)
summary(out.log.vs)


# 5-fold cross-validation to estimate test error ####
misc<-function(yhat,y){
  a<-table(yhat,y)
  1-sum(diag(a))/sum(a)
}

K<-5
n<-nrow(SAheart)
set.seed(15)
index<-sample(1:K,n,replace=TRUE)
table(index)
x<-subset(SAheart,select=(c("chd","tobacco","ldl","famhist","typea","age")))
yy.hat<-rep(NA,n)
CV.err<-NULL

for (i in 1:K){
  x.test<-x[index==i,]
  x.train<-x[index!=i,] # all the units for which index!=i
  y.test<-x$chd[index==i]
  
  out.log.cv<-glm(chd~.,data=x.train,family="binomial")
  p.hat<-predict(out.log.cv,newdata=x.test,type="response")
  y.hat<-ifelse(p.hat>=0.5,1,0)
  CV.err[i]<-misc(y.hat,y.test)
  yy.hat[index==i]<-y.hat
}
mean(CV.err)
table(SAheart$chd,yy.hat)

##

library(ROCR)
out.log.cv
pred<-prediction(p.hat, labels = y.test)
sens.spec<-performance(pred,"sens","spec")
par(mfrow=c(1,1))
plot(sens.spec, colorize = TRUE)
auc<-performance(pred,"auc")
auc@y.values
