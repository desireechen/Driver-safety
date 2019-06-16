train.data <- read.csv("traindata.csv")
validation.data <- read.csv("validationdata.csv")
# Prepare training data
df.training <- data.frame(x1=train.data[ , 3], x2=train.data[ , 4], y=train.data$label)
summary(df.training)
df.training$y <- as.factor(df.training$y)
# Prepare validation data
df.grid <- data.frame(x1=validation.data[,3], x2=validation.data[,4])
df.grid$prob <- validation.data[,5]
# Visualise data
library("ggplot2")
p0 <- ggplot() + geom_point(data=df.training, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw()
p0

# Cross-Validation parameters
K <- 10  # 10-fold CV
RUN <- 10  # the number of repetitions of CV
I <- 9  # the max polynomical order to consider
# Prepare data frame to store error values.
err.kfold <- expand.grid(run=1:RUN, i=1:I)
err.kfold$err <- 0
# Misclassification error with cutoff = 0.5
cost <- function(y, p = 0) {mean(y != (p > 0.5))}

## Start cross-validation
library("boot")
set.seed(987)
for (run in 1:RUN) {
  for (i in 1:I) {
    glm.fitted <- glm(y ~ poly(x1, i) + poly(x2, i), family=binomial(), data=df.training)
    err.kfold[err.kfold$run == run & err.kfold$i == i, "err"] <- cv.glm(df.training, glm.fitted, cost, K=K)$delta[1]  # specify K=10 for k-fold CV
  }
}
head(err.kfold)
summary(err.kfold)
# Plot the error for each run and each i
library("ggplot2")
ggplot(data=err.kfold, aes(x=i, y=err)) + geom_line(aes(color=factor(run))) + scale_x_discrete () + theme_bw()
# To find the run and i with the minimum err
err.kfold[err.kfold$err == min(err.kfold$err),]
# To see the CV error for i = 6
err.kfold[err.kfold$i == 6,]
# To get the mean of the CV errors for all the 10 runs
mean(err.kfold[err.kfold$i == 6,]$err)

## Manual k-fold Cross Validation
err.kfold2 <- expand.grid(run=1:RUN, i=1:I)
err.kfold2$err <- 0
set.seed(987)
for (run in 1:RUN) {
  # Create a random partition. Compare the 10 models on the same data.
  folds <- sample(1:K, nrow(df.training),replace=TRUE)
  for (i in 1:I) {
    err <- 0  # To store overall misclassfication errors in all k folds
    # start the k-fold CV. In the training data, it is everything except the kth fold.
    for (k in 1:K) {
      ### fit a glm on everything except the kth fold
      glm.fitted <- glm(y ~ poly(x1, i) + poly(x2, i), family=binomial(), data=df.training[folds != k, ])
      ### use the fitted glm to predict on the kth fold
      glm.prob <- predict(glm.fitted, newdata=df.training[folds == k, ], type="response")
      glm.pred <- ifelse(glm.prob > 0.5, 1, 0)
      glm.err <- sum(glm.pred != df.training[folds == k, "y"])
      err <- err + glm.err
    }
    err.kfold2[err.kfold2$run == run & err.kfold2$i == i, "err"] <- err / nrow(df.training)
  }
}
head(err.kfold2)
ggplot(data=err.kfold2, aes(x=i, y=err)) + geom_line(aes(color=factor(run))) + scale_x_discrete () + theme_bw()
# To find the run and i with the minimum err
err.kfold2[err.kfold2$err == min(err.kfold2$err),]
# To see the CV error for i = 7
err.kfold2[err.kfold2$i == 7,]
# To get the mean of the CV errors for all the 10 runs
mean(err.kfold2[err.kfold2$i == 7,]$err)

# Get ROC plots of manual k-fold Cross Validation models
library("ROCR")
i <- 4
set.seed(987)
for (run in 1:RUN) {
  # create a random partition
  folds <- sample(1:K, nrow(df.training), replace=TRUE)
  # start the k-fold CV
  for (k in 1:K) {
    ### fit a glm on everything except the kth fold
    glm.fitted <- glm(y ~ poly(x1, i) + poly(x2, i), family=binomial(), data=df.training[folds != k, ])
    df.training[folds == k, "prob"] <- predict(glm.fitted, newdata=df.training[folds == k, ], type="response")
  }
  pred <- prediction(df.training$prob, df.training$y)
  perf <- performance(pred, measure="tpr", x.measure="fpr")
  plot(perf, add=(run != 1))
  auc <- as.numeric(performance(pred, "auc")@y.values) 
}
auc