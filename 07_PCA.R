# Lab: Unsupervised Learning


## Principal Components Analysis

###
states <- row.names(USArrests)
states
###
names(USArrests)
###
apply(USArrests, 2, mean)
###
apply(USArrests, 2, var)
###
pr.out <- prcomp(USArrests, scale = TRUE)
###
names(pr.out)
###
pr.out$center#means
pr.out$scale#standard deviations
###
pr.out$sdev#L (sqrt of the eigenvalues of cov(X))
pr.out$rotation##V (eigenvectors of cov(X))
###
pr.out$x##Z=UD=XV
###
par(mfrow=c(1,1))
biplot(pr.out, scale = 0)
###
pr.out$rotation = -pr.out$rotation
pr.out$x = -pr.out$x
biplot(pr.out, scale = 0)
###
pr.out$sdev
###
pr.var <- pr.out$sdev^2
pr.var
###
pve <- pr.var / sum(pr.var)
pve
###
par(mfrow = c(1, 2))
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 1),
     type = "b")
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")
###
a <- c(1, 2, 8, -3)
cumsum(a)
###

## Singular Value Decomposition

x.std<-scale(USArrests,T,T) # standardize the data
svd_x<-svd(x.std)
loadings_svd<-as.matrix(svd_x$v)
rownames(loadings_svd)<-colnames(USArrests)
colnames(loadings_svd)<-c("PC1","PC2","PC3","PC4")
loadings_svd
pr.out$rotation
pc2<-svd_x$u[,1:2]%*%diag(svd_x$d)[1:2,1:2]
pr.out$x[,1:2]
##

rownames(pc2)<-rownames(USArrests)
par(mfrow=c(1,1))
plot(pc2,pch=20,xlab="PC1",ylab="PC2")
abline(h=0,v=0,lty=2)
text(pc2, labels=rownames(pc2))
points(pc2[c("California","Mississippi"),],col="blue",pch=2)
loadings_svd