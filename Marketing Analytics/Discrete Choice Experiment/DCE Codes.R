setwd("C:/Users/isabe/Desktop/MSDS 450/Solo 2/Solo2 Codes")
require(dummies)
setwd("C:/Users/BDR0TYX/Downloads")
load("stc-cbc-respondents-v3.RData")
load("efCode.RData")
taskV3 <- read.csv("stc-dc-task-cbc -v3.csv", sep="\t")
task.mat <- as.matrix(taskV3[, c("screen", "RAM", "processor", "price", "brand")])
X.mat=efcode.attmat.f(task.mat)
pricevec=taskV3$price-mean(taskV3$price)
X.brands=X.mat[,9:11]
X.BrandByPrice = X.brands*pricevec
X.matrix=cbind(X.mat,X.BrandByPrice)
det(t(X.matrix)%*%X.matrix)
load("stc-cbc-respondents-v3.RData")
ydata=resp.data.v3[,4:39]
names(ydata)
ydata=na.omit(ydata)
ydata=as.matrix(ydata)
zowner <- 1 * ( ! is.na(resp.data.v3$vList3) )
require(bayesm)
lgtdata = NULL
for (i in 1:424) {
  	   lgtdata[[i]]=list(y=ydata[i,],X=X.matrix)
}
table(ydata)
colSums(X.matrix)

require(knitr)
df_X.corr <- cor(X.matrix)
kable(X.matrix)

sum(lgtdata[[12]]$y)
table(lgtdata[[35]]$y)

length(lgtdata)
lgtdata[[3]]
str(lgtdata)
###########################################################################
# Model Evaluation
###########################################################################

#-------------------------------------------------------------------------
# Model 1
#-------------------------------------------------------------------------

# Specify iterations
#mcmctest <- list(R=5000, keep=5) # run 5,000 iterations and keep every 5th
help(package="bayesm")
lgtdata100=lgtdata[1:100]
mcmctest=list(R=30000,keep=5)

R <- 30000 #5000
keep <- 10 #5
ndraws <- R/keep
rm(R, keep)
# Create data input list
Data1=list(p=3,lgtdata=lgtdata)
# p is the number of choice models, fed to rhierMnlDP
# note that X is subdivided by p in order to match y, i.e. 108 / 3 = 36

# Test run
testrun1=rhierMnlDP(Data=Data1,Mcmc=mcmctest)
#names(testrun1)

dim(testrun1$betadraw)

# Get betadraw
betadraw1=testrun1$betadraw
dim(betadraw1)
# 424 rows (case/respondent), 14 columns (beta estimate), n draws (where n is based on R / keep)
plot(1:length(betadraw1[1,1,]),betadraw1[1,1,])
plot(density(betadraw1[1,1,701:1000],width=2))
summary(betadraw1[1,1,701:1000])
apply(betadraw1[,,701:1000],c(2),mean)
apply(betadraw1[,,701:1000],c(1,2),mean)
summary((betadraw1[1,1,701:1000]-betadraw1[1,2,701:1000]))
plot(density(betadraw1[1,1,701:1000]-betadraw1[1,2,701:1000],width=2))
betameansoverall <- apply(betadraw1[,,701:1000],c(2),mean)
betameansoverall
perc <- apply(betadraw1[,,701:1000],2,quantile,probs=c(0.05,0.10,0.25,0.5 ,0.75,0.90,0.95))
perc

zownertest=matrix(scale(zowner,scale=FALSE),ncol=1)
Data2=list(p=3,lgtdata=lgtdata,Z=zownertest)
testrun2=rhierMnlDP(Data=Data2,Mcmc=mcmctest)
dim(testrun2$deltadraw)
apply(testrun2$Deltadraw[701:1000,],2,mean)





