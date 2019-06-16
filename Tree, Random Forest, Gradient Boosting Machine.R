train.data <- read.csv("traindata.csv")
validation.data <- read.csv("validationdata.csv")

# Prepare training and validation data
train.data$label <- as.factor(train.data$label)
validation.data$label <- as.factor(validation.data$label)
x.test <- validation.data[, c(3:4)]
y.test <- validation.data[, 5]

## Tree model
library("tree")
tree1 <- tree(label ~ avgspeed + avgrest, data=train.data)
tree1
summary(tree1)
# Plot tree
plot(tree1)
text(tree1, pretty=TRUE, cex=0.8) 
# Conduct cross validationn on the tree size
set.seed(987)
tree1.cv <- cv.tree(tree1, method="misclass")
tree1.cv
plot(tree1.cv)
optimal <- which.min(tree1.cv$dev)
optimal.k <- tree1.cv$k[optimal]
optimal.size <- tree1.cv$size[optimal]
# Prune tree
tree1.pruned <- prune.tree(tree1, best=optimal.size, method="misclass")
tree1.pruned
plot(tree1.pruned)
text(tree1.pruned, pretty=TRUE)
prob.tree1.pruned <- predict(tree1.pruned, newdata=validation.data, type="vector")[,2]
head(prob.tree1.pruned)
summary(prob.tree1.pruned)

## Random Forest
library("randomForest")
err.rate.rfs <- rep(0, 13)
set.seed(987) 
for(m in 1:13){
  rf <- randomForest(label ~ avgspeed + avgrest, data=train.data, ntree=500, mtry=m)
  err.rate.rfs[m] <- rf$err.rate[500]
}
plot(1:13, err.rate.rfs, type="b", xlab="mtry", ylab="OOB Error")
optimal.mtry <- which.min(err.rate.rfs)
optimal.OOBerror <- err.rate.rfs[which.min(err.rate.rfs)]
# Build model
set.seed(987)
data.rf2 <- randomForest(label ~ ., data=train.data, mtry=optimal.mtry, ntree=500)
data.rf2
plot(data.rf2)
# Use fitted model to predict
prob.rf2 <- predict(data.rf2, newdata=validation.data, type="prob")[,2]
head(prob.rf2)
summary(prob.rf2)
pred.rf2 <- predict(data.rf2, newdata=validation.data, type="response")
head(pred.rf2)
table(pred.rf2, y.test)

## Gradient Boosting Machine
library("gbm")
train.data$label <- as.numeric(train.data$label)-1
set.seed(987)
data.gbm <- gbm(label ~ ., data=train.data, distribution="bernoulli", n.trees=4000, shrinkage=0.001, interaction.depth=6)
data.gbm
prob.data.gbm <- predict(data.gbm, newdata=validation.data, n.trees=4000, type="response")
head(prob.data.gbm)
summary(prob.data.gbm)
pred.gbm <- as.factor(ifelse(prob.data.gbm > 0.5, "Yes", "No"))
head(pred.gbm)
summary(pred.gbm)
table(pred.gbm, y.test)

## Plot ROC curves
library("ROCR")
pred3.tree1.pruned <- prediction(prob.tree1.pruned, y.test)
pred3.rf <- prediction(prob.rf2, y.test) 
pred3.gbm <- prediction(prob.data.gbm, y.test)
err.tree1 <- performance(pred3.tree1.pruned, measure="err")
err.rf <- performance(pred3.rf, measure="err")
err.gbm <- performance(pred3.gbm, measure="err")
plot(err.tree1, ylim=c(0.1,0.9))
plot(err.rf, col="red", add=TRUE)
plot(err.gbm, col="blue", add=TRUE)
roc.tree1 <- performance(pred3.tree1.pruned, measure="tpr", x.measure="fpr")
roc.rf <- performance(pred3.rf, measure="tpr", x.measure="fpr")
roc.gbm <- performance(pred3.gbm, measure="tpr", x.measure="fpr")
plot(roc.tree1)
plot(roc.rf, col="red", add=TRUE)
plot(roc.gbm, col="blue", add=TRUE)
abline(a=0, b=1, lty=2) # diagonal line
as.numeric(performance(pred3.tree1.pruned, "auc")@y.values)
as.numeric(performance(pred3.rf, "auc")@y.values) 
as.numeric(performance(pred3.gbm, "auc")@y.values) 