
setwd("C:/Users/Syamala.srinivasan/Google Drive/NorthWestern/Predict450/RProjects")

load("apphappyData.RData")
ls()
## [1] "apphappy.2.labs.frame" "apphappy.2.num.frame" ##

#Library will load the existing loaded package. 
#Require will install or update when the package is not in our repository

require(cluster)
require(useful)
require(Hmisc)
require(plot3D)
library(HSAUR)
library(MVA)
library(HSAUR2)
library(fpc)
library(mclust)
library(lattice)
library(car)
library(digest)

numdata <- apphappy.2.num.frame
str(numdata)


### Creating subsets ###

numsubr <- subset(numdata, select=
                   c("q24r1","q24r2","q24r3","q24r4","q24r5","q24r6","q24r7","q24r8","q24r9",
                     "q24r10","q24r11","q24r12",
                     "q25r1","q25r2","q25r3","q25r4","q25r5","q25r6","q25r7","q25r8","q25r9",
                     "q25r10","q25r11","q25r12",
                     "q26r3","q26r4","q26r5","q26r6","q26r7","q26r8","q26r9","q26r10","q26r11",
                     "q26r12","q26r13","q26r14","q26r15","q26r16","q26r17","q26r18"))

numsub <- subset(numdata, select=
c("q24r1","q24r2","q24r3","q24r4","q24r5","q24r6","q24r7","q24r8","q24r9",
  "q24r10","q24r11","q24r12",
  "q25r1","q25r2","q25r3","q25r4","q25r5","q25r6","q25r7","q25r8","q25r9",
  "q25r10","q25r11","q25r12",
  "q26r3","q26r4","q26r5","q26r6","q26r7","q26r8","q26r9","q26r10","q26r11",
  "q26r12","q26r13","q26r14","q26r15","q26r16","q26r17","q26r18"))


str(numsub)
head(numsub)
attach(numsub)


#######################
#### correlation plot##
#######################
require(corrplot)
mcor <- cor(numsub)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)
summary(numsub)
#######################################################
### check for peaks & valleys (ie) natural clusters ###
#######################################################

##  Create cuts:
q24r1_c <- cut(q24r1, 6)
q24r2_c <- cut(q24r2, 6)

##  Calculate joint counts at cut levels:
z <- table(q24r1_c, q24r2_c)
z

##  Plot as a 3D histogram:
hist3D(z=z, border="black")

##  Plot as a 2D heatmap:

image2D(z=z, border="black")

library(latticeExtra)

cloud(z~q24r1_c+q24r2_c, numsub, panel.3d.cloud=panel.3dbars, col.facet='blue', 
      xbase=0.4, ybase=0.4, scales=list(arrows=FALSE, col=1), 
      par.settings = list(axis.line = list(col = "transparent")))

#######################################
############### PCA Plots ##############
######################################
dev.off()
pca <-princomp(numsub)
plot(pca$scores[,1],pca$scores[,2])

names(pca)
str(pca)
summary(pca)
head(pca$scores)

sort(pca$scores[,1])

numsub["2367",]
numsub["1307",]
numsub["552",]
numsub["2261",]

sort(pca$scores[,1], decreasing = TRUE)

numsub["1904",]
numsub["243",]


##  Create cuts:
pcadf <- as.data.frame(pca$scores)
pca1 <- cut(pcadf$Comp.1, 10)
pca2 <- cut(pcadf$Comp.2, 10)

##  Calculate joint counts at cut levels:
z <- table(pca1, pca2)

##  Plot as a 3D histogram:
hist3D(z=z, border="black")


###################################################
### Create a 'scree' plot to determine the num of clusters
#####################################################

dev.off()
wssplot <- function(numsub, nc=15, seed=1234) {
  wss <- (nrow(numsub)-1)*sum(apply(numsub,2,var))
  for (i in 2:nc) {
    set.seed(seed)
    wss[i] <- sum(kmeans(numsub, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")} 

wssplot(numsubr)

#######################################################
##########  k means with raw data with 5 clusters######
#######################################################

clusterresults <- kmeans(numsubr,5)
clusterresults$size
rsquare <- clusterresults$betweenss/clusterresults$totss
rsquare
str(clusterresults)
plot(clusterresults, data=numsubr)
dev.off()
dissE <- daisy(numsubr)
names(dissE)
dE2   <- dissE^2
sk2   <- silhouette(clusterresults$cluster, dE2)
str(sk2)

plot(sk2)

############################################################
### Clustering for Likert scale - 'ordinal' data
### 40 variables with 6 values (ie) 240 binary columns
### For each pair of people, compute % match on 240 columns
### Similarity metric is this % & do clustering
### For Solo 1 we will use derived variable concept
### (ie) combining columns and 'pretend' continous data
### then use clustering methods for the continuous data

#############################################################
##create 'derived' variables - means of similar variables ###
#############################################################

attach(numsub)
numsub$q24a <- (q24r1+q24r2+q24r3+q24r5+q24r6)/5
numsub$q24b <- (q24r7+q24r8)/2
numsub$q24c <- (q24r10+q24r11)/2
numsub$q24d <- (q24r4+q24r9+q24r12)/3

numsub$q25a <- (q25r1+q25r2+q25r3+q25r4+q25r5)/5
numsub$q25b <- (q25r7+q25r8)/2
numsub$q25c <- (q25r9+q25r10+q25r11)/3
numsub$q25d <- (q25r6+q25r12)/2

numsub$q26a <- (q26r3+q26r4+q26r5+q26r6+q26r7)/5
numsub$q26b <- (q26r8+q26r9+q26r10)/3
numsub$q26c <- q26r11
numsub$q26d <- (q26r12+q26r13+q26r14)/3
numsub$q26e <- (q26r15+q26r16+q26r17+q26r18)/4

numsub2 <- subset(numsub, select=
                   c("q24a","q24b","q24c",
                     "q25a","q25b","q25c",
                     "q26a","q26b","q26d","q26e"))

pca <-princomp(numsub2)
plot(pca$scores[,1],pca$scores[,2])
names(pca)
head(pca$scores)
str(pca$scores)
summary(pca)

pcadf <- as.data.frame(pca$scores)
pca1 <- cut(pcadf$Comp.1, 10)
pca2 <- cut(pcadf$Comp.2, 10)

##  Calculate joint counts at cut levels:
z <- table(pca1, pca2)

##  Plot as a 3D histogram:
hist3D(z=z, border="black")

require(corrplot)
mcor <- cor(numsub2)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=0.5)

#######################################################
### Create a Kmeans with 5 clusters with derived data #
#######################################################
clusterresults <- kmeans(numsub2,5)
names(clusterresults)
rsquare <- clusterresults$betweenss/clusterresults$totss
rsquare
plot(clusterresults, data=numsub2)
dev.off()
dissE <- daisy(numsub2)
names(dissE)
dE2   <- dissE^2
sk2   <- silhouette(clusterresults$cluster, dE2)
str(sk2)
plot(sk2)

##### another way to do the same thing ################

newdf <- as.data.frame(clusterresults$cluster)
pcadf <- as.data.frame(pca$scores)

write.csv(newdf, file = "clusterresults.csv")
write.csv(pcadf, file = "pca.csv")
combdata <- cbind(newdf,pcadf)
head(combdata)

xyplot(Comp.2 ~ Comp.1, combdata, groups = clusterresults$cluster, pch= 20)

write.csv(numsub, file = "numsub.csv")

################################################################
### Create a dataset with the original data with the cluster info
### This will be useful for creating profiles for the clusters
###############################################################

newdf <- read.csv("clusterresults.csv")
combdata <- cbind(numsub2,newdf,numdata$q1)
head(combdata)

require(reshape)
combdata <- rename(combdata, c(clusterresults.cluster="cluster"))
head(combdata)

aggregate(combdata,by=list(byvar=combdata$cluster), mean)
## Done with K Means, do the profile  ###
###############################################
## Hierarchical Clustering with derived data ##
###############################################
numsub.dist = dist(numsub2)
require(maptree)
hclustmodel <- hclust(dist(numsub2), method = 'ward.D2')
names(hclustmodel)
plot(hclustmodel)

cut.5 <- cutree(hclustmodel, k=5)
plot(silhouette(cut.5,numsub.dist))
head(cut.5)

########################################
##for hclust how to calculate BSS & TSS
######################################
require(proxy)
numsubmat <- as.matrix(numsub2)
overallmean <- matrix(apply(numsubmat,2,mean),nrow=1)
overallmean
TSS <- sum(dist(numsubmat,overallmean)^2)
TSS
####################################
#Compute WSS based on 5 clusters
######################################
combcutdata <- cbind(numsub2,cut.5)
head(combcutdata)

require(reshape)
combcutdata <- rename(combcutdata, c(cut.5="cluster"))
head(combcutdata)

clust1 <- subset(combcutdata, cluster == 1)
clust1 <- subset(clust1, select=c("q24a","q24b","q24c","q25a","q25b","q25c",
                                  "q26a","q26b","q26d","q26e"))
clust1 <- as.matrix(clust1,rowby=T)
dim(clust1)
clust1mean <- matrix(apply(clust1,2,mean),nrow=1)
dim(clust1mean)
dis1 <- sum(dist(clust1mean,clust1)^2)

clust2 <- subset(combcutdata, cluster == 2)
clust2 <- subset(clust2, select=c("q24a","q24b","q24c","q25a","q25b","q25c",
                                  "q26a","q26b","q26d","q26e"))
clust2 <- as.matrix(clust2,rowby=T)
clust2mean <- matrix(apply(clust2,2,mean),nrow=1)
dis2 <- sum(dist(clust2mean,clust2)^2)

clust3 <- subset(combcutdata, cluster == 3)
clust3 <- subset(clust3, select=c("q24a","q24b","q24c","q25a","q25b","q25c",
                                  "q26a","q26b","q26d","q26e"))
clust3 <- as.matrix(clust3,rowby=T)
clust3mean <- matrix(apply(clust3,2,mean),nrow=1)
dis3 <- sum(dist(clust3mean,clust3)^2)

clust4 <- subset(combcutdata, cluster == 4)
clust4 <- subset(clust4, select=c("q24a","q24b","q24c","q25a","q25b","q25c",
                                  "q26a","q26b","q26d","q26e"))
clust4 <- as.matrix(clust4,rowby=T)
clust4mean <- matrix(apply(clust4,2,mean),nrow=1)
dis4 <- sum(dist(clust4mean,clust4)^2)

clust5 <- subset(combcutdata, cluster == 5)
clust5 <- subset(clust5, select=c("q24a","q24b","q24c","q25a","q25b","q25c",
                                  "q26a","q26b","q26d","q26e"))
clust5 <- as.matrix(clust5,rowby=T)
clust5mean <- matrix(apply(clust5,2,mean),nrow=1)
dis5 <- sum(dist(clust5mean,clust5)^2)

WSS <- sum(dis1,dis2,dis3,dis4,dis5)
WSS

BSS <- TSS - WSS
BSS
## calculating the % of Between SS/ Total SS
rsquare <- BSS/TSS
rsquare


#######################################################
### A little function to calculate the average silhouette width
### for a variety of choices of k:
###########################################################
my.k.choices <- 2:8
avg.sil.width <- rep(0, times=length(my.k.choices))
for (ii in (1:length(my.k.choices)) ){
  avg.sil.width[ii] <- pam(numsub2, k=my.k.choices[ii])$silinfo$avg.width
}
print( cbind(my.k.choices, avg.sil.width) )
#################################
# PAM method
###############################
clusterresultsPAM <-pam(numsub2,5)
summary(clusterresultsPAM)
str(clusterresultsPAM$silinfo)
plot(clusterresultsPAM, which.plots=1)
plot(clusterresultsPAM, which.plots=2)

###################
## Model based clustering
##################
library(mclust)
fit <- Mclust(numsub2,5)
plot(fit,data=numsub2, what="density") # plot results
#plot(fit,data=numsub2, what="BIC") # plot results

summary(fit) # display the best model

dev.off()
dissE <- daisy(numsub2)
names(dissE)
dE2   <- dissE^2
sk2   <- silhouette(fit$classification, dE2)
str(sk2)
plot(sk2)

###############################################
## comparison of cluster results  #############
###############################################
clstat <- cluster.stats(numsub.dist, clusterresults$cluster, clusterresultsPAM$cluster)
names(clstat)
clstat$corrected.rand
##corrected or adjusted rand index lies between 0 & 1
## perfect match between 2 clustering methods means 1, no match means 0
## any number in between represents 'kind of' % of matched pairs 

clstat <- cluster.stats(numsub.dist, clusterresults$cluster, cut.5)
clstat$corrected.rand


















