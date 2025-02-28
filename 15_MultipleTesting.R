########################################################
# generate data:
########################################################
# artificially generate 10,000 p-values
n_0 = 8000
n_1 = 2000

# 800 from the null:
null_p = runif(n_0)
hist(null_p)

# 200 from the alternative:
alt_p = rbeta(n_1, 1, 50)
hist(alt_p, xlim = c(0, 1))

# merge p-values (keep track of truth, not used)
DF = data.frame(p_values = c(null_p, alt_p), 
                truth = c(rep(0, n_0),
                          rep(1, n_1)))
head(DF); tail(DF)
hist(DF$p_values)

########################################################
# adjust p-values
########################################################
?p.adjust
DF$FWER = p.adjust(DF$p_values, method = "bonferroni")
DF$FDR = p.adjust(DF$p_values, method = "BH")

# Cut raw and adjusted p-values at threshold alpha:
alpha = 0.1
cut_raw  = DF$p_values < alpha
cut_FWER = DF$FWER < alpha
cut_FDR  = DF$FDR < alpha

# FWER -> very conservative:
hist(DF$FWER)
# p-value < alpha/m   <=>   p-value * m < alpha
# therefore p-values are multiplied by m (and thresholded if > 1).
head(sort(DF$FWER), 20)
(n_0 + n_1) * head(sort(DF$p_values), 20)

# FDR -> also conservative, but less extreme
hist(DF$FDR)

########################################################
# study FPs:
########################################################
sum(DF$truth[cut_raw] == 0)
# expected:
n_0 * alpha

sum(DF$truth[cut_FWER] == 0)
# expected:
alpha

sum(DF$truth[cut_FDR] == 0)
# expected?
# not known -> FDR refers to the set of significant results!
# among the significant results, we expect 10% to be FPs

# let's look at the significant results:
DF$truth[cut_FDR]
# we counts what fraction of these are FPs:
mean(DF$truth[cut_FDR] == 0)
# on average we expect:
alpha

# very useful piece of information!
# we have 
sum(cut_FDR)
# significant results;
# we expect 90% of these to be true discoveries.

########################################################
# study TPs:
########################################################
# in all cases, no expected number of TP detections
# -> distribution of p-values under H1 not known

sum(DF$truth[cut_raw] == 1)
# most TPs have a p-value < alpha

sum(DF$truth[cut_FWER] == 1)
# no expected -> distribution of p-values under H1 not known

sum(DF$truth[cut_FDR] == 1)
# among the significant results, we expect 90% to be TPs!

# let's look at the FDR significant results:
DF$truth[cut_FDR]
# we counts what fraction of these are TPs:
mean(DF$truth[cut_FDR] == 1)
# on average we expect:
1-alpha

########################################################
# ranking not altered (top results are the same):
########################################################
# same result gives the minimum raw and adjusted p-value
which.min(DF$p_values)
which.min(DF$FWER)
which.min(DF$FDR)

# ROC curve of the 3 should be identical
# it may not be though: why?
library(pROC)
p_values=roc(DF$truth, DF$p_values)
FWER=roc(DF$truth, DF$FWER)
FDR=roc(DF$truth, DF$FDR)
library(ggplot2)
ggroc(list(p_values = p_values,
           FWER = FWER,
           FDR = FDR))
# ROC curves based on raw p-values and BH-adjusted p-values are the same
ggroc(list(p_values = p_values))
ggroc(list(FDR = FDR))

# ROC curve based on Bonferroni-adjusted p-values is very different though:
ggroc(list(FWER = FWER))

# this is because most Bonferroni-adjusted p-values are thresholded at 1
# it is an artifact
head(sort(DF$FWER), 20)
