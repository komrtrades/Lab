# Clustering ####

## k-means ####
### Generate synthetic data from Gaussian distribution and
### apply a shift.
set.seed(1234)
x<-matrix(rnorm(50*2), ncol=2) # units are from the same popul.

plot(x, main="", xlab="", ylab="", pch=20, cex=2)

# apply a shift on the first 25 units, 1st variable
x[1:25, 1]<-x[1:25, 1]+3 
# apply a shift on the first 25 units, 2nd variable
x[1:25, 2]<-x[1:25, 2]-4
y<-rep(1:2, each=25) # vector of cluster membership

plot(x, main="", xlab="", ylab="", pch=20, cex=2)

# K-means
# K = 2 clusters
??kmeans
km.out<-kmeans(x, 2, nstart = 20)
km.out$cluster # the vector of labels
table(km.out$cluster, y)

plot(x, col=(km.out$cluster+1), 
     main="K-means clustering results with K=2", 
     xlab="", ylab="", pch=20, cex=2)

# K = 3 clusters
set.seed(1234)
km.out3<-kmeans(x, 3, nstart=20)
km.out3
table(km.out3$cluster, y)
# The additional cluster is obtained by
# taking 3 units from cluster 2 and 7 units from cluster 1

plot(x, col=(km.out3$cluster+1), 
     main="K-means clustering results with K=3", 
     xlab="", ylab="", pch=20, cex=2)

# drawback: k-means will identify 3 plausible clusters, 
# even though data was generated from 2 only.

# nstart defines the number of times we start the algorithm
# increasing nstart allows to decrease the within SS
set.seed(17)
km.out<-kmeans(x, 3, nstart=1)
km.out$tot.withinss
km.out$cluster

km.out2<-kmeans(x, 3, nstart=20)
km.out2$tot.withinss # we improved the results by running the algo 20 times
km.out2$cluster



# also visually (results differ with nstart=1).
# due to the seed
par(mfrow = c(1, 2))
plot(x, col=kmeans(x, 3, nstart=1)$cluster+1, 
     main="K-means clustering results with K=3", 
     xlab="", ylab="", pch=20, cex=2)


# increasing nstart=20 gives the same result
# even with different seeds (more stable results)
plot(x, col=kmeans(x, 3, nstart=20)$cluster+1, 
     main="K-means clustering results with K=3", 
     xlab="", ylab="", pch=20, cex=2)

par(mfrow = c(1, 1))

### Choosing K via the Elbow Methods ####
set.seed(123)
# Compute and plot wss for k = 1 to k = 10.
k.max <- 10
wss <- sapply(1:k.max, 
              function(k){
                kmeans(x, k, nstart=50, iter.max = 100 )$tot.withinss
              })
wss
plot(1:k.max, wss, 
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K", 
     ylab="Total within-clusters sum of squares")

# probably K = 2, or 3 at most.

### Choosing K via Gap Statistic ####
# gap statistic is a measure of how much 
# higher the within-cluster SS is for a given k
# than it is for a hypothetical reference distribution
# that captures the clustering structure given by the data.

#install.packages("cluster")
library(cluster)
set.seed(1234)
gs.x<-clusGap(x, kmeans, K.max=10, B=1000)
gs.x

plot(gs.x, main="Gap Statistic")
# Gap suggests to choose K = 2 (4-5 quire close)
# Are we confident about this result?

set.seed(1)
gs.x<-clusGap(x, kmeans, K.max=10, B=100)
plot(gs.x, main="Gap Statistic")

set.seed(2)
gs.x<-clusGap(x, kmeans, K.max=10, B=100)
plot(gs.x, main="Gap Statistic")

set.seed(3)
gs.x<-clusGap(x, kmeans, K.max=10, B=100)
plot(gs.x, main="Gap Statistic")

# Note that the procedure is noisy and, changing the seed, leads to different results!

# How can we decrease the noise?
# Increasing B?
set.seed(1)
gs.x<-clusGap(x, kmeans, K.max=10, B=10^3)
plot(gs.x, main="Gap Statistic")

set.seed(2)
gs.x<-clusGap(x, kmeans, K.max=10, B=10^3)
plot(gs.x, main="Gap Statistic")

set.seed(3)
gs.x<-clusGap(x, kmeans, K.max=10, B=10^3)
plot(gs.x, main="Gap Statistic")

# NO: kmeans is computes using nstart = 1.
# the noise comes from the kmeans
# we increase nstart to 20
set.seed(1)
gs.x<-clusGap(x, kmeans, K.max=10, B=100, nstart = 20)
plot(gs.x, main="Gap Statistic")

set.seed(2)
gs.x<-clusGap(x, kmeans, K.max=10, B=100, nstart = 20)
plot(gs.x, main="Gap Statistic")

set.seed(3)
gs.x<-clusGap(x, kmeans, K.max=10, B=1000, nstart = 20)
plot(gs.x, main="Gap Statistic")
# This leads to more stable results
# The Gap Statistic suggests to use K = 2

### Measuring clustering quality ####
# we compute 3 metrics of the clustering quality
# for varying k 
# install.packages('fpc')
library(fpc)

# compute k-means from 2 to 8 clusters:
km2<-kmeans(x, 2, nstart=20)
km3<-kmeans(x, 3, nstart=20)
km4<-kmeans(x, 4, nstart=20)
km5<-kmeans(x, 5, nstart=20)
km6<-kmeans(x, 6, nstart=20)
km7<-kmeans(x, 7, nstart=20)
km8<-kmeans(x, 8, nstart=20)

# Compute dissimilarity between observatioons:
dim(x); head(x)
?dist
d.x<-dist(x) # Euclidean dissimilarity matrix between rows of x
str(d.x)
length(d.x);
# lower triangular part of the 50 * 50 matrix (excluding the diagonal):
# 50 (1st obs) * 49 (2nd obs)/2
50 * 49 / 2

?cluster.stats
# Compute Cluster validation statistics for various K:
out2<-cluster.stats(d.x, km2$cluster)
out3<-cluster.stats(d.x, km3$cluster)
out4<-cluster.stats(d.x, km4$cluster)
out5<-cluster.stats(d.x, km5$cluster)
out6<-cluster.stats(d.x, km6$cluster)
out7<-cluster.stats(d.x, km7$cluster)
out8<-cluster.stats(d.x, km8$cluster)

str(out2)
# a lot of information: we extract the 3 scores we are interested in

out.m<-matrix(NA, 7, 3, dimnames = list(
  c(paste("K=", 2:8)), c("ASW", "PG", "CH")))
out.m[1, ]<-c(out2$avg.silwidth, out2$pearsongamma, out2$ch)
out.m[2, ]<-c(out3$avg.silwidth, out3$pearsongamma, out3$ch)
out.m[3, ]<-c(out4$avg.silwidth, out4$pearsongamma, out4$ch)
out.m[4, ]<-c(out5$avg.silwidth, out5$pearsongamma, out5$ch)
out.m[5, ]<-c(out6$avg.silwidth, out6$pearsongamma, out6$ch)
out.m[6, ]<-c(out7$avg.silwidth, out7$pearsongamma, out7$ch)
out.m[7, ]<-c(out8$avg.silwidth, out8$pearsongamma, out8$ch)

# 3 statistics:
out.m
# In general, higher values = better clustering. 
# In this case, k = 2 is the best option.

# we look for K which max each statistics: 1st row (k=2)
apply(out.m, 2, which.max)

# K = 2 in all cases.
rownames(out.m)[apply(out.m, 2, which.max)]
# Each clustering quality criterion selects K=2 as best option.


## Hierarchical Clustering ####
?hclust
# Hierarchical Clustering is performed on a dissimilarity matrix
hc.complete<-hclust(d.x, method="complete")
hc.average<-hclust(d.x, method="average")
hc.single<-hclust(d.x, method="single")

hc.complete$merge # illustrates the pairs (of observations or cluster) merges performed at each step
hc.complete$height # includes the corresponding dissimilarity of the
# two clusters (observations) merged at each step

par(mfrow=c(1, 3)) # split the plot window in 3 parts
plot(hc.complete, main="Complete Linkage", xlab="", 
     sub="", cex=0.9)
plot(hc.average, main="Average Linkage", xlab="", 
     sub="", cex=0.9)
plot(hc.single, main="Single Linkage", xlab="", 
     sub="", cex=0.9)

# we cut each tree with 2 branches:
cl.complete<-cutree(hc.complete, k=2)
cl.average<-cutree(hc.average, k=2)
cl.single<-cutree(hc.single, k=2)

# we compare results with real separation (unknown to the clustering):
table(cl.complete, y) # good
table(cl.average, y) # perfect
table(cl.single, y) # terrible: it only separates 1 observation

# We subtract the mean from each column, and make variance unitary
# sometimes normalizing data can improve results
?scale
x.std<-scale(x)
apply(x, 2, mean); apply(x.std, 2, mean); 
apply(x, 2, sd); apply(x.std, 2, sd); 
par(mfrow = c(1,2))
plot(x, main="", xlab="", ylab="", pch=20, cex=2)
plot(x.std, main="", xlab="", ylab="", pch=20, cex=2)
# only axes change

hc.complete.std<-hclust(dist(x.std), method="complete")
dev.off()
plot(hc.complete.std, main="Hierarchical clustering 
with scaled features") 

cl.complete.std<-cutree(hc.complete.std, 2)
table(cl.complete.std, y) # perfect separation
# scaling features has improved the resulting clustering

hc.average.std<-hclust(dist(x.std), method="average")
cl.average.std<-cutree(hc.average.std, 2)
table(cl.average.std, y) # perfect separation (results for average linkage didnt change)


#install.packages("mclust") # for AdjustedRandIndex
# adjustedRandIndex = index to compare how similar 2 sets are:
library(mclust)
?adjustedRandIndex
# The adjusted Rand index (ARI) is the corrected-for-chance version of the Rand index.
# The Rand Index computes a similarity measure between two clusterings.
# Though the (unadjusted) Rand Index may only yield a value between 0 and +1, 
# the adjusted Rand index can yield negative values if the index is less than the expected index.
# The ARI index has zero expected value in the case of random partition, 
# and it is bounded above by 1 in the case of perfect agreement between two partitions.
adjustedRandIndex(cl.complete, y)
adjustedRandIndex(cl.average, y)
adjustedRandIndex(cl.single, y)
adjustedRandIndex(cl.complete.std, y)

# draw a partition at random:
rbinom(n = length(y), size = 1, prob = 0.5)
adjustedRandIndex(rbinom(n = length(y), size = 1, prob = 0.5), 
                  y)
# close to 0

#### Correlation-based similarity ####
# we could also base our dissimilarity on the correlation
# between pairs of observations
# however, to do this, we need at least 3 variables:

cor(t(x))[1:5, 1:5]
# If p=2, correlation matrix will always be equal to 1.
# => if p=2 you can't use correlation as similarity measure.
# correlation between pairs of observations
# each observation has p = 2 variables:
# there will always be a line passing though 2 points
# so corr = 1

# Let's create a new dataset with p=3
set.seed(1234)
xx<-matrix(rnorm(30*3), ncol=3)

# Dissimilarity-based correlation can be computed
# in two ways:
# *with 1-cor: objects with cor=-1 will be considered
#  very different
# *with 1-abs(cor): objects with cor=-1 will be
#  considered as perfectly overlapping, no distance

# Compute the distance matrix as as.dist(1-cor)
# high correlations btw observations -> they are similar
# so dissimilarity = 1 - cor
dd<-as.dist(1-cor(t(xx)))
# We perform clustering based on the distance matrix
hc.cor<-hclust(dd, method="complete")
plot(hc.cor, main="Complete Linkage with 
     Correlation-based distance")


# NCI60 Data example ####
# install.packages('ISLR')
library(ISLR)
help(NCI60)
# NCI microarray data. 
# The data contains expression levels on 
# 6830 genes from 64 cancer cell lines.
# Cancer type is also recorded.

nci.data<-NCI60$data
nci.labs<-NCI60$labs

dim(nci.data)
# n = 64
# p = 6830
table(nci.labs)

## PCA on NCI60 Data ####
?prcomp
pc.out<-prcomp(nci.data, scale=TRUE)
summary(pc.out)

# contribution of each component:
plot(pc.out)

# simple function to color cancer lines differently
Cols<-function(vec){
  cols=rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}

# plot components by pairs:
plot(pc.out$x[, 1:2], col=Cols(nci.labs), 
     pch=19, xlab="Z1", ylab="Z2")
# colours represent tissue:
unique(nci.labs)
# if we want to explore more?
# 3D plot or more pairwise 2D plots.

# components 1 vs 3:
plot(pc.out$x[, c(1, 3)], col=Cols(nci.labs), 
     pch=19, xlab="Z1", ylab="Z3")

# alternative way to plot PCA:
# autoplot authomatically adds the percentage of variance explained by each PC
# install.packages("ggfortify")
library(ggfortify)
autoplot(pc.out, col= Cols(nci.labs))
# ~18% of variance of the data in only 2 components...
# ...out of p = ~7 k variables: impressive summary!

# To reduce the dimensionality, we can select a
# smaller number of PCs to perform clustering on:
plot(pc.out)

# the % of variance explained by each PCs, and cumulative %
summary(pc.out)$importance[, 1:6]
pve = summary(pc.out)$importance[2,]
head(pve)

plot(pve, type="o", ylab="PVE", xlab="Principal Component")
abline(v = 7)
plot(cumsum(pve), type="o", ylab="Cumulative PVE", xlab="Principal Component")
abline(v = 7)

# Clustering on the first 7 PCs 
sum(pve[1:7])
# ~ 40% of the variance in the data explained
nci.pc<-pc.out$x[, 1:7]
d.pc<-dist(nci.pc) # dissimilarity matrix computed on the PCs
hc.complete.pc<-hclust(d.pc, method="complete")
hc.average.pc<-hclust(d.pc, method="average")
hc.single.pc<-hclust(d.pc, method="single")

par(mfrow=c(1, 3))
plot(hc.complete.pc, labels=nci.labs, 
     main="Complete Linkage", xlab="", ylab="", sub="")
plot(hc.average.pc, labels=nci.labs, 
     main="Average Linkage", xlab="", ylab="", sub="")
plot(hc.single.pc, labels=nci.labs, 
     main="Single Linkage", xlab="", ylab="", sub="")

hc.pc.4<-cutree(hc.complete.pc, 4)
ha.pc.4<-cutree(hc.average.pc, 4)
hs.pc.4<-cutree(hc.single.pc, 4)

table(hc.pc.4, nci.labs)
table(ha.pc.4, nci.labs)
table(hs.pc.4, nci.labs)
# very hard to compare to the real labels (too many: 14 in total)

# What about kmeans?
km.out<-kmeans(nci.pc, 4, nstart=20)
table(km.out$cl, nci.labs)

# quite hard to establish quality of clustering from the tables above.
# below we compute some metrics.

### Evaluate clutering quality ####
library(fpc)
?cluster.stats # Computes a number of distance based statistics
# d: a distance object; in our case the dissimilarity matrix
# clustering: an integer vector which indicates a clustering

quality.hc<-cluster.stats(d.pc, hc.pc.4)
quality.ha<-cluster.stats(d.pc, ha.pc.4)
quality.hs<-cluster.stats(d.pc, hs.pc.4)
quality.km<-cluster.stats(d.pc, km.out$cl)

str(quality.hc)
# various metrics; we consider:
# avg.silwidth
# pearsongamma
# ch: Calinski and Harabasz index

quality<-matrix(NA, 4, 3, dimnames = 
                  list(c("HC-Complete", 
                         "HC-Average", 
                         "HC-Single",
                         "K-means"), 
                       c("ASW", "PG", "CH")))
quality[1, ]<-c(quality.hc$avg.silwidth, 
                quality.hc$pearsongamma, quality.hc$ch)
quality[2, ]<-c(quality.ha$avg.silwidth, 
                quality.ha$pearsongamma, quality.ha$ch)
quality[3, ]<-c(quality.hs$avg.silwidth, 
                quality.hs$pearsongamma, quality.hs$ch)
quality[4, ]<-c(quality.km$avg.silwidth, 
                quality.km$pearsongamma, quality.km$ch)
quality
# in all 3 measures: high values = good classification
# they don't agree in this case:
# ASW -> average and single link
# PG -> average and single link
# CH -> k-means

### How well do the clustering methods recover the true cluster membership? ####

# Adjusted Rand Index
library(mclust)
?adjustedRandIndex
# Computes the adjusted Rand index comparing two classifications.
adjustedRandIndex(hc.pc.4, nci.labs)
# 0.1927713
adjustedRandIndex(ha.pc.4, nci.labs)
# 0.03738577
adjustedRandIndex(hs.pc.4, nci.labs)
# 0.03738577
adjustedRandIndex(km.out$cl, nci.labs)
# 0.2220284
# highest ARI -> k-means and complete link


## Clustering the original Data (INSTEAD OF PCA!) ####
sd.data<-scale(nci.data) # standardize the original data

data.dist<-dist(sd.data) # Euclidean distance matrix

hc.sd<-hclust(data.dist, method="complete")
ha.sd<-hclust(data.dist, method="average")
hs.sd<-hclust(data.dist, method="single")

par(mfrow=c(1, 3))
plot(hc.sd, labels=nci.labs, main="Complete Linkage", xlab="", ylab="", sub="")
plot(ha.sd, labels=nci.labs, main="Average Linkage", xlab="", ylab="", sub="")
plot(hs.sd, labels=nci.labs, main="Single Linkage", xlab="", ylab="", sub="")

### Cut the trees so as to have 4 clusters ####
hc.cl<-cutree(hc.sd, 4)
ha.cl<-cutree(ha.sd, 4)
hs.cl<-cutree(hs.sd, 4)

adjustedRandIndex(hc.cl, nci.labs)
# 0.1267859
adjustedRandIndex(ha.cl, nci.labs)
# 0.0501931
adjustedRandIndex(hs.cl, nci.labs)
# -0.002517982
# still complete link, but much lower ARI comped to using PCs

quality.hc.cl<-cluster.stats(data.dist, hc.cl)
quality.hc.cl$avg.silwidth
# 0.07021168
quality.hc.cl$pearsongamma
# 0.5634049

# previous values, based on PCs:
quality.hc$avg.silwidth 
# 0.2541779
quality.hc$pearsongamma
# 0.5257471

# Similarly, also avg.silwidth decreases significantly.
# pearsongamma: not all metrics point at the same direction.

# take home message: with many predictors,
# using PCs lead to significantly better results, 
# compared to using the original data.

### Compare with k-means, with K=4 #####
km.sd<-kmeans(sd.data, 4, nstart=20)
table(km.sd$cluster, nci.labs)

adjustedRandIndex(km.sd$cluster, nci.labs)
# same ARI as before:
adjustedRandIndex(km.out$cluster, nci.labs)

# k-means is more robust: similar ARI with both input data.

