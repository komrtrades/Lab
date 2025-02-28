## MARCO Example with public dataset
# 1. Load dataset
data(mtcars)
mtcars$am <- as.factor(mtcars$am)  # Transmission: 0 = automatic, 1 = manual
mtcars$cyl <- as.factor(mtcars$cyl)  # Number of cylinders as a factor
mtcars$gear <- as.factor(mtcars$gear)  # Number of gears as a factor
mtcars$carb <- as.factor(mtcars$carb)  # Number of carburetors as a factor

# 2. Split dataset into training and test sets
set.seed(123)  # For reproducibility
unita.totali <- 1:nrow(mtcars)
unita.training <- sample(x=unita.totali, size=0.75*nrow(mtcars))
unita.test <- unita.totali[-unita.training]
training.set <- mtcars[unita.training, ]
test.set <- mtcars[unita.test, ]

# 3. Linear regression
fit1 <- lm(mpg ~ ., data=training.set)
summary(fit1) 
# Remove one variable for comparison
fit2 <- lm(mpg ~ . - gear, data=training.set)
# 4. Compare the two models using ANOVA
anova(fit1, fit2, test="Chisq")

# 5. Assessing predictive capability
# Confusion matrix and misclassification rate for fit2
predict2 <- predict(fit2, newdata=test.set, type='response')
Y.prev <- as.integer(predict2 > 0.5)  # Bayes rule

# Build confusion matrix
mat.conf <- table(Y.prev, test.set$am)

# Misclassification rate
t.err.class <- 1 - (sum(diag(mat.conf)) / sum(mat.conf))
print(t.err.class)

# 6. Lift curve
library(ROCR)
pred <- prediction(predict2, labels = test.set$am)
perf <- performance(pred, "lift", "rpp")
plot(perf, main="Lift Curve", colorize=TRUE)

# 7. Calculate lift manually
prob.ord <- sort(predict2, decreasing=TRUE)
soglie <- quantile(prob.ord, probs = seq(from=0.99, to=0.01, by=-0.1))
lift <- NULL
for(i in 1:length(soglie)){
  Y.prev <- as.integer(predict2 > soglie[i])
  mat.conf <- table(Y.prev, test.set$am)
  lift[i] <- (mat.conf[2,2] / (mat.conf[2,1] + mat.conf[2,2])) / 
             ((mat.conf[1,2] + mat.conf[2,2]) / sum(mat.conf))
}
plot(seq(from=0.99, to=0.01, by=-0.1), rev(lift), type="l", ylab= "Lift", xlab="Quantile")

# 8. Sensitivity and specificity
sens <- mat.conf[2,2] / (mat.conf[2,2] + mat.conf[1,2])
spec <- mat.conf[1,1] / (mat.conf[1,1] + mat.conf[2,1])
print(sens)
print(spec)

# 9. ROC curve
sens.spec <- performance(pred, "sens", "spec")
plot(sens.spec, colorize = TRUE)

###################################################################################################



# 1. load dataset
juice<-read.table("./data/juice.txt",header=T)
juice$scelta<-as.factor(juice$scelta)
juice$negozio <- as.factor(juice$negozio)
names(juice)<-c("choice","ID.customer","week","priceCH","priceMM","discountCH","discountMM","loyalCH","loyalMM","shop")

# 2. estimation
set.seed(123) # replicability 
unita.totali<-1:1070
unita.training<-sample(x=unita.totali,size=0.75*1070)
unita.test<-unita.totali[-unita.training]
training.set<-juice[unita.training,]
test.set<-juice[unita.test,]

fit1<-glm(choice ~ week + priceCH + priceMM + discountCH + discountMM + loyalMM + shop, data=training.set, family=binomial)
summary(fit1)

# 3. remove week
fit2<-glm(choice ~ priceCH + priceMM + discountCH + discountMM + loyalMM + shop, data=training.set, family=binomial)
summary(fit2)

# (test anova)
anova(fit1,fit2,test="Chisq")

# 4. Assessing predictive capability
# Confusion matrix and misclassification rate
predict2<-predict.glm(fit2, newdata = test.set, type = 	'response')
Y.prev<-as.integer(predict2 > 0.5) # Bayes rule
t.err.class<- 1-(sum(diag(mat.conf)/sum(mat.conf)))

# lift curve
library(ROCR)
pred<-prediction(predict2, labels = test.set[, 1]) 
perf <- performance(pred,"lift","rpp")
plot(perf, main="lift curve", colorize=T)

# or by hand:
prob.ord <- sort(predict2, decreasing = T)
soglie <- quantile(prob.ord, probs = seq(from=0.99, to=0.01, by=-0.1))
lift <- NULL
for(i in 1:length(soglie)){
  Y.prev<-as.integer(predict2 > soglie[i]) # Bayes rule
  mat.conf<-table(Y.prev, test.set[, 1]) # build the confusion matrix
  lift[i]<-((mat.conf[2,2])/(mat.conf[2,1]+mat.conf[2,2]))/((mat.conf[1,2]+mat.conf[2,2])/sum(mat.conf))
}
plot(seq(from=0.99, to=0.01, by=-0.1), rev(lift), type="l", ylab= "lift", xlab="quantile")

# sensitivity and specificity
sens<-mat.conf[2,2]/(mat.conf[2,2] + mat.conf[1,2]) 
spec<-mat.conf[1,1]/(mat.conf[1,1] + mat.conf[2,1]) 
# ROC curve
sens.spec<-performance(pred,"sens","spec")
plot(sens.spec, colorize = TRUE)
