## Practical Machine Learning Assignment
## Support script
## 15.09.26

## setup
library(knitr)
library(RCurl)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
set.seed(1111)
#opts_chunk$set(echo = TRUE, message = FALSE, error = TRUE, warning = TRUE, comment = NA, fig.align = "center", dpi = 100, cache = TRUE, tidy = FALSE, cache.path = ".cache/", fig.path = "fig/")
#setwd("C:\\data\\synched\\local\\trabalho\\pessoais\\formação\\2015\\15.06 Data Science Specialization\\08 Practical Machine Learning\\assignment 1")

## rawdata
setInternet2(use = TRUE)
if (!file.exists("data/pml-training.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="data/pml-training.csv")
}
if (!file.exists("data/pml-testing.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="data/pml-testing.csv")
}
train <- read.csv("data/pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test  <- read.csv("data/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
#head(train); head(test)
dim(train); dim(test)

## tidydata
train <- train[,colSums(is.na(train)) == 0] # 100 columns with all values NA
train <- train[rowSums(is.na(train)) == 0,] # no rows with all values NA
train <- train[c(-1)]
trainnzv <- nearZeroVar(train, saveMetrics=TRUE)
train <- train[!trainnzv$nzv]
dim(train)
cleantrain <- colnames(train)
cleantest <- colnames(train[, -58])
test <- test[cleantest]
dim(test)
message("Near zero attributes removed:")
trainnzv[trainnzv$nzv,]

## tidydata2
for (i in 1:length(test) ) {
        if(class(test[,i]) != class(train[,i])) {
                message(sprintf("Column %i '%s' is a %s on the test dataframe but it is a %s in the train dataframe.", i, colnames(test[i]), class(test[,i]), class(train[,i])))
        }
}
test <- rbind(train[2, -58] , test)
test <- test[-1,]


## crossvalidation
filter <- createDataPartition(y=train$classe, p=0.95, list=FALSE)
trainsub <- train[filter, ] 
traintest <- train[-filter, ]
dim(trainsub); dim(traintest)

## histogram
plot(trainsub$classe,
     main="No. of observations in each class",
     xlab="Classes",
     ylab="No. of observations")
sum(trainsub$classe == "A")
sum(trainsub$classe == "D")

## tree
treemodel <- rpart(classe ~ ., data=trainsub, method="class")
prediction <- predict(treemodel, traintest, type = "class")
#rpart.plot(treemodel, main="Classification Tree", extra=102, under=TRUE, faclen=0)
fancyRpartPlot(treemodel)
confusionMatrix(prediction, traintest$classe)

## randomforest
rfmodel <- randomForest(classe ~ . , data=trainsub)
rfprediction <- predict(rfmodel, traintest, type="class")
confusionMatrix(rfprediction, traintest$classe)

## comparing
tc <- trainControl(method = "cv", number = 7, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
tree <- train(classe ~ ., data = train, method = "M5", trControl= tc)
rf <- train(classe ~ ., data = train, method = "rf", trControl= tc)
svmr <- train(classe ~ ., data = train, method = "svmRadial", trControl= tc)
NN <- train(classe ~ ., data = train, method = "nnet", trControl= tc, verbose=FALSE)
svml <- train(classe ~ ., data = train, method = "svmLinear", trControl= tc)
bayesglm <- train(classe ~ ., data = train, method = "bayesglm", trControl= tc)
logitboost <- train(classe ~ ., data = train, method = "LogitBoost", trControl= tc)
## Loading required package: caTools
Accuracy comparision

library(RWeka)
library(nnet)
model <- c("Tree", "Random Forest", "SVM (radial)","LogitBoost","SVM (linear)","Neural Net", "Bayes GLM")
Accuracy <- c(max(tree$results$Accuracy),
              max(rf$results$Accuracy),
              max(svmr$results$Accuracy),
              max(logitboost$results$Accuracy),
              max(svml$results$Accuracy),
              max(NN$results$Accuracy),
              max(bayesglm$results$Accuracy))

Kappa <- c(max(tree$results$Kappa),
           max(rf$results$Kappa),
           max(svmr$results$Kappa),
           max(logitboost$results$Kappa),
           max(svml$results$Kappa),
           max(NN$results$Kappa),
           max(bayesglm$results$Kappa))  

performance <- cbind(model,Accuracy,Kappa)

## output
finalprediction <- predict(rfmodel, test, type="class")
for(i in 1:length(finalprediction)) {
        #filename = paste0("output/problem_id_",i,".txt")
        #write.table(finalprediction[i], file=filename, quot=FALSE, row.names=FALSE, col.names=FALSE)
        message(sprintf("Prediction for problem %i: %s", i, finalprediction[i]))
}
message(finalprediction)
