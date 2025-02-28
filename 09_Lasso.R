## LASSO path for logistic regression
rm(list=ls())

## SAheart dataset
library(bestglm)
data("SAheart")
summary(SAheart)
str(SAheart)

##Validation set approach
set.seed(15)
n<-nrow(SAheart)
ind_val<-sample(1:n,ceiling(n/4),replace=F)
ind_tr<-setdiff(1:n,ind_val)

##necessary to estimate the model
#SAheart$chd<-as.factor(SAheart$chd)
SAheart$famhist<-as.numeric(SAheart$famhist)-1
##

##create partitions
SAheart.train.val<-as.matrix(SAheart[ind_tr,])
SAheart.test.val<-as.matrix(SAheart[ind_val,])
##

##allocate x and y
y.train.val<-SAheart.train.val[,10]
x.train.val<-SAheart.train.val[,-10]
y.test.val<-SAheart.test.val[,10]
x.test.val<-SAheart.test.val[,-10]
##

##glmpath
data(SAheart)
attach(SAheart)
str(SAheart)
library(glmpath)
?glmpath
SA.train.out <- glmpath(x.train.val, y.train.val, family=binomial)
print(SA.train.out)
SA.train.cv.out <- cv.glmpath(x.train.val, y.train.val, family="binomial")
SA.train.summary.out <- summary.glmpath(SA.train.out)
SA.train.summary.out
plot.glmpath(SA.train.out)
##

## estimate the reduced logistic model
colnames(x.train.val)
x.top.val<-as.data.frame(x.train.val[,-c(1,4,7,8)])
out.top<-glm(y.train.val~.,data=x.top.val,family="binomial")
#heart.test<-as.data.frame(heart.data$x[ind_val,-c(4,7,8)])
summary(out.top)
names(x.top.val)
out.top.restr<-glm(y.train.val~.,data=x.top.val[,-c(1,4)],family="binomial")
summary(out.top.restr)
anova(out.top,out.top.restr)
qchisq(0.95,2)
qchisq(0.99,2)

## estimation error in the test set
pred_val<-predict.glm(out.top.restr, newdata = as.data.frame(x.test.val),type = 'response')
chd.pred.val<-as.integer(pred_val > 0.5) 
mat.conf<-table(chd.pred.val, y.test.val) 
t.err.val<- 1-(sum(diag(mat.conf)/sum(mat.conf)))
t.err.val
#detach(heart.data)
##
