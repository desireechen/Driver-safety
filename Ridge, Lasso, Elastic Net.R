train.data <- read.csv("traindata.csv")
validation.data <- read.csv("validationdata.csv")

# Prepare training and validation data
train.data <- train.data[,-c(1:2)]
validation.data <- validation.data[,-c(1:2)]
# Make data into a matrix form. Package "glmnet" requires data in matrix form.
x <- model.matrix(label ~ ., train.data)[,-1]
y <- train.data$label
y <- as.factor(y)
x.test <- model.matrix(label ~ ., validation.data)[,-1]
y.test <- validation.data$label
y.test <- as.factor(y.test)

##  Build a model using glm
model.full <- glm(label ~ ., data=train.data, family=binomial())
summary(model.full)
model.null <- glm(label ~ 1, data=train.data, family=binomial())
summary(model.null)
# Use stepAIC to find the best model
library("MASS")
model1 <- stepAIC(model.full, direction="both", scope=list(upper=model.full, lower=model.null))
summary(model1)
model1.prob <- predict(model1, newdata=validation.data, type="response")
head(model1.prob)
summary(model1.prob)

## Ridge Regression
library("glmnet")
model.ridge <- glmnet(x, y, family="binomial", alpha=0)
set.seed(987)
ridge.cv <- cv.glmnet(x, y, family="binomial", type.measure="class", alpha=0)
ridge.lam <- ridge.cv$lambda.min
# Use lambda.1se as the optimal lambda for a more parsimonious model
ridge.lam2 <- ridge.cv$lambda.1se
ridge.lam2
ridge.prob <- predict(model.ridge, newx=x.test, s=ridge.lam2, family="binomial", type="response", exact=TRUE) 
head(ridge.prob)
summary(ridge.prob)

## LASSO
model.lasso <- glmnet(x, y, family="binomial", alpha=1)
set.seed(987)
lasso.cv <- cv.glmnet(x, y, family="binomial", type.measure="class", alpha=1)
lasso.lam <- lasso.cv$lambda.min
lasso.lam2 <- lasso.cv$lambda.1se
lasso.lam2
lasso.prob <- predict(model.lasso, newx=x.test, s=lasso.lam2, family="binomial", type="response", exact=TRUE) 
head(lasso.prob)
summary(lasso.prob)

## Elastic Net
K <- 10
n <- nrow(x)
fold <- rep(0, n)
set.seed(987)
shuffled.index <- sample(n, n, replace=FALSE)
fold[shuffled.index] <- rep(1:K, length.out=n)
table(fold) # each of the datapoints is going into a chunk
fold # to see which chunk each of the 13990 data points goes to
alphas <- seq(0, 1, 0.1)
en2.cv.error <- data.frame(alpha=alphas)
for (i in 1:length(alphas)){
  en2.cv <- cv.glmnet(x, y, alpha=alphas[i], family="binomial", type.measure="class", foldid=fold)
  en2.cv.error[i, "lambda.min"] <- en2.cv$lambda.min
  en2.cv.error[i, "error.min"] <- min(en2.cv$cvm)
  en2.cv.error[i, "lambda.1se"] <- en2.cv$lambda.1se
  en2.cv.error[i, "error.1se"] <- min(en2.cv$cvm) + en2.cv$cvsd[which.min(en2.cv$cvm)]
}
en2.cv.error
en2.lam2 <- en2.cv.error[which.min(en2.cv.error$error.1se), "lambda.1se"]
en2.lam2
en2.alpha2 <- en2.cv.error[which.min(en2.cv.error$error.1se), "alpha"]
en2.alpha2
en2.mod <- glmnet(x, y, family="binomial", alpha=en2.alpha2)
en2.prob <- predict(en2.mod, newx=x.test, s=en2.lam2, family="binomial", type="response", exact=TRUE)
head(en2.prob)
summary(en2.prob)

## Get predictions of each model.
model1.pred2 = ifelse(model1.prob > 0.5, 1, 0)
ridge.pred2 = ifelse(ridge.prob > 0.5, 1, 0)
lasso.pred2 = ifelse(lasso.prob > 0.5, 1, 0)
en2.pred2 = ifelse(en2.prob > 0.5, 1, 0)
# Create confusion mat for each model
confusion.mat.model1 <- table(model1.pred2, y.test)
misclass.model1 <- (confusion.mat.model1[2, 1] + confusion.mat.model1[1, 2]) / nrow(x.test)
misclass.model1
confusion.mat.ridge <- table(ridge.pred2, y.test)
misclass.ridge <- (confusion.mat.ridge[2, 1] + confusion.mat.ridge[1, 2]) / nrow(x.test)
misclass.ridge
confusion.mat.lasso <- table(lasso.pred2, y.test)
misclass.lasso <- (confusion.mat.lasso[2, 1] + confusion.mat.lasso[1, 2]) / nrow(x.test)
misclass.lasso
confusion.mat.en2 <- table(en2.pred2, y.test)
misclass.en2 <- (confusion.mat.en2[2, 1] + confusion.mat.en2[1, 2]) / nrow(x.test)
misclass.en2
# Plot ROC curves
library("ROCR")
pred.model1 <- prediction(model1.prob, y.test)
pred.ridge <- prediction(ridge.prob, y.test)
pred.lasso <- prediction(lasso.prob, y.test)
pred.en2 <- prediction(en2.prob, y.test)
err.model1 <- performance(pred.model1, measure="err")
err.ridge <- performance(pred.ridge, measure="err")
err.lasso <- performance(pred.lasso, measure="err")
err.en2 <- performance(pred.en2, measure="err")
plot(err.model1, col="darkorchid1", ylim=c(0.1,0.9))
plot(err.ridge, col="red", add=TRUE)
plot(err.lasso, col="blue", add=TRUE)
plot(err.en2, col="black", add=TRUE)
ROC.model1 <- performance(pred.model1, measure="tpr", x.measure="fpr")
ROC.ridge <- performance(pred.ridge, measure="tpr", x.measure="fpr")
ROC.lasso <- performance(pred.lasso, measure="tpr", x.measure="fpr")
ROC.en2 <- performance(pred.en2, measure="tpr", x.measure="fpr")
plot(ROC.model1, col="darkorchid1")
plot(ROC.ridge, col="red", add=TRUE)
plot(ROC.lasso, col="blue", add=TRUE)
plot(ROC.en2, col="black", add=TRUE)
abline(a=0, b=1, lty=2) # diagonal line
# Get AUC for each model
as.numeric(performance(pred.model1, "auc")@y.values) 
as.numeric(performance(pred.ridge, "auc")@y.values) 
as.numeric(performance(pred.lasso, "auc")@y.values) 
as.numeric(performance(pred.en2, "auc")@y.values) 