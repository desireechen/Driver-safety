## Part 1: Work on the Test Data first.
# Load required libraries.
library(plyr)
library(dplyr)
# Load test data. Please rename test file to mytestdata.csv
test.data <- read.csv("mytestdata.csv")
# Sort test data.
test.ordered <- test.data[with(test.data, order(bookingID, second)), ]
# Create new variables.
test.ordered$racc <- sqrt(test.ordered$acceleration_x^2+test.ordered$acceleration_y^2+test.ordered$acceleration_z^2)
w <- 9
test.ordered$rxest <- (test.ordered$acceleration_x+(test.ordered$gyro_x*w))/(1+w)
test.ordered$ryest <- (test.ordered$acceleration_y+(test.ordered$gyro_y*w))/(1+w)
test.ordered$rzest <- (test.ordered$acceleration_z+(test.ordered$gyro_z*w))/(1+w)
test.ordered$rest <- 10*(sqrt(test.ordered$rxest^2+test.ordered$ryest^2+test.ordered$rzest^2))
# Remove data that has second equals to 0.
test.ordered <- test.ordered[!(test.ordered$second==0),]
# For each bookingID, summarise speed and estimated R. Also, get the label column.
trips.summary <- ddply(test.ordered, .(bookingID), summarize,  avgspeed=mean(Speed), avgrest=mean(rest), label=mean(label))

## Part 2: Prepare the training and validation data for building the model.
# Read label data.
labels <- read.csv("part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
labels.no.duplicates <- labels[-c(12464,12603,2352,5296,6213,11216,6122,19937,8473,
                                  17624,2859,10779,11059,18028,16463,17844,10881,17002,
                                  11134,18952,2722,13635,6169,6211,3069,16991,9980,
                                  14943,9484,16341,1257,6517,1060,14433,13689,19454),] 
df.labels.no.duplicates <- as.data.frame(labels.no.duplicates)
# Read feature data.
feature0 <- read.csv("part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature1 <- read.csv("part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature2 <- read.csv("part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature3 <- read.csv("part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature4 <- read.csv("part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature5 <- read.csv("part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature6 <- read.csv("part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature7 <- read.csv("part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature8 <- read.csv("part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
feature9 <- read.csv("part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
allfeature <- rbind(feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9)
df.allfeature <- as.data.frame(allfeature)
# Combine label and feature data.
df <- left_join(df.allfeature, df.labels.no.duplicates, by="bookingID")
# Remove NAs.
df.no.NAs <- na.omit(df)
# Make label a factor.
df.no.NAs$label <- as.factor(df.no.NAs$label)
# Sort data.
df.ordered <- df.no.NAs[with(df.no.NAs, order(bookingID, second)), ]
# Create new variables.
df.ordered$racc <- sqrt(df.ordered$acceleration_x^2+df.ordered$acceleration_y^2+df.ordered$acceleration_z^2)
w <- 9
df.ordered$rxest <- (df.ordered$acceleration_x+(df.ordered$gyro_x*w))/(1+w)
df.ordered$ryest <- (df.ordered$acceleration_y+(df.ordered$gyro_y*w))/(1+w)
df.ordered$rzest <- (df.ordered$acceleration_z+(df.ordered$gyro_z*w))/(1+w)
df.ordered$rest <- 10*(sqrt(df.ordered$rxest^2+df.ordered$ryest^2+df.ordered$rzest^2))
# Extract only relevant columns to build model.
df.ordered <- df.ordered[,-c(2,3,4,5,6,7,8,9,13,14,15,16)]
# Remove data that has second equals to 0.
df.ordered <- df.ordered[!(df.ordered$second==0),]
# For each bookingID, summarise speed and estimated R.
trips.summary <- ddply(df.ordered, .(bookingID), summarize,  avgspeed=mean(Speed), avgrest=mean(rest))
# For building the model, we have the label column. The test data will not have label column.
trips.with.labels <- left_join(trips.summary, df.labels.no.duplicates, by="bookingID")
trips.ordered <- trips.with.labels[with(trips.with.labels, order(label)), ]
# Scale data.
trips.ordered$avgspeed <- (trips.ordered$avgspeed - mean(trips.ordered$avgspeed))/(max(trips.ordered$avgspeed)-min(trips.ordered$avgspeed))
trips.ordered$avgrest <- (trips.ordered$avgrest - mean(trips.ordered$avgrest))/(max(trips.ordered$avgrest)-min(trips.ordered$avgrest))
# Separate data into training set and validation set.
train0 <- trips.ordered[c(1:10500),]
train1 <- trips.ordered[c(15000:18489),]
test0 <- trips.ordered[c(10501:14999),]
test1 <- trips.ordered[c(18490:19982),]
train.data <- rbind(train0, train1)
train.data$label <- as.factor(train.data$label)
validation.data <- rbind(test0, test1)
validation.data$label <- as.factor(validation.data$label)
total.data <- rbind(train.data, validation.data)
total.data$label <- as.factor(total.data$label)
# Building the model.
model.full <- glm(label ~ ., data=total.data, family=binomial())
summary(model.full)
model.null <- glm(label ~ 1, data=total.data, family=binomial())
summary(model.null)
library("MASS")
model1 <- stepAIC(model.full, direction="both", scope=list(upper=model.full, lower=model.null))
summary(model1)

## Part 3: Use fitted model to predict on the test data.
model1.prob <- predict(model1, newdata=trips.summary, type="response")
head(model1.prob)
summary(model1.prob)
library("ROCR")
y.val <- trips.summary$label
y.val <- as.factor(y.val)
pred.model1 <- prediction(model1.prob, y.val)
err.model1 <- performance(pred.model1, measure="err")
plot(err.model1, col="blue", ylim=c(0.1,0.9))
ROC.model1 <- performance(pred.model1, measure="tpr", x.measure="fpr")
plot(ROC.model1, col="blue")
abline(a=0, b=1, lty=2) # diagonal line
as.numeric(performance(pred.model1, "auc")@y.values)