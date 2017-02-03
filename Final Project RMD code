---
title: "FinaL_machine_learning_project"
author: "Devender Singh Saini"
date: "February 2, 2017"
output: html_document
---

# Introduction
In this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.

Below is the code for data cleaning and predicting the output

# Load used packages
```{r message=FALSE}
library(caret)
library(plyr)
library(dplyr)
library(randomForest)
library(xgboost)
library(knitr)

```


# Data loading and cleaning
```{r}
train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(train_url),na.strings = c("NA","#DIV/0!",""),header = T)
testing <- read.csv(url(test_url),na.strings = c("NA","#DIV/0!",""),header = T)
dim(training)  # dimension of training dataset
dim(testing)   # dimension of testing dataset
training <- training[,-c(1)]  # remove index
testing <- testing[,-c(1)]  # remove index
```



# Make training and validation data
```{r}
set.seed(123)
index <- createDataPartition(training$classe,p=0.6,list = F)
train_tr <- training[index,]    # training dataset
train_val <- training[-index,]  # validation dataset
dim(train_tr)
dim(train_val)

```

# Remove zerovar variables
Remove variables which has zero varaition in dataset from both training and validation dataset
```{r}
nsv <- nearZeroVar(train_tr,saveMetrics = T)  
train_tr <- train_tr[,nsv$nzv==FALSE]

nsv <- nearZeroVar(train_val,saveMetrics = T)  
train_val <- train_val[,nsv$nzv==FALSE]
```


# Removing columns with greater than 60% NA values
varna variable below is dataframe in which 1st row is column name and 2nd row is how much percentage of NAs in that columns. Using for loop, we are finding columns with less than 60% NA's
```{r}
var60 <- list()  # finding columns where NA values are less than 60%
varna <- train_tr %>% summarise_each(funs(sum(is.na(.))*100/dim(train_tr)[1])) 
for (i in names(train_tr))
{  if (any(varna[i] < 60)){
       var60 <- c(var60,i)   
}
}
length(var60) # no of column with NA values less than 60%

train_tr <- train_tr[,c(unlist(var60))]  # taking columns with less than 60% NA
dim(train_tr)
var60[58]<- NULL   # remove "classe" column
length(var60)
testing <- testing[,c(unlist(var60))] # taking columns with less than 60% NA
dim(testing)

```

# Set levels equal in training and testing data
Sometimes, if levels are not equal in training and testing data then while doing prediction error comes. So, it is always better to set same levels for both the datasets.
```{r}
common <- intersect(names(train_tr), names(testing)) 
for (p in common) { 
  if (class(train_tr[[p]]) == "factor") { 
    levels(testing[[p]]) <- levels(train_tr[[p]]) 
  } 
}
```


# Predictive model
## Random Forest
1) Fit the model
2) Predict the output based on validation data
3) Check the accuracy
```{r}
set.seed(123)
model_rf <- randomForest(classe~.,data = train_tr)
pred_rf <- predict(model_rf,train_val)
confusionMatrix(pred_rf, train_val$classe)  # accuray: 99.91
plot(model_rf)
varImpPlot(model_rf)  # Can check which variable is contributing most in the prediction(most important)

```


# Generalized boosting models
Here I have done cross validation with 10 folds
```{r message=FALSE}
set.seed(123)
fitcontrol <- trainControl(method = "repeatedcv",number = 10,repeats = 2)
model_gbm <- train(classe~.,data = train_tr,method='gbm',trControl=fitcontrol,verbose=F)
pred_gbm <- predict(model_gbm,train_val)
confusionMatrix(pred_gbm, train_val$classe)  # accuracy: 99.76
plot(model_gbm)

```


# Extreme Gradient Boosting(Xgboost)
Here I have used Xgboost method. Generally, it gives very high accuracy,but here in this model, it is giving bad results. While making the final predictions of testing data, I am not using this model.
```{r}
set.seed(123)
levels(train_tr$classe)[1] <- "0"   # conveting dependent variable
levels(train_tr$classe)[2] <- "1"
levels(train_tr$classe)[3] <- "2"
levels(train_tr$classe)[4] <- "3"
levels(train_tr$classe)[5] <- "4"

train_tr_y <- train_tr$classe
train_tr$classe <- NULL
dmy <- dummyVars("~ .",data = train_tr)  # xgboost requires all variables to be numeric(one hot encoding)
train_tr_dum <- data.frame(predict(dmy,newdata=train_tr))

cv.res1 <- xgb.cv(data = as.matrix(train_tr_dum), label = as.matrix(train_tr_y), nfold = 10,
                  nrounds = 300, objective = "multi:softmax", num_class = 5,eta = 0.1,
                  eval_metric = 'merror',early_stopping_round = 40,missing = missing,verbose = F)  # to find the optimal no of trees

# optimal tress comes out to be 230. Use this to predict
bst1 <- xgboost(data = as.matrix(train_tr_dum), label = as.matrix(train_tr_y),nrounds =230 , 
                objective = "multi:softmax", num_class = 5,eta = 0.1,verbose = F)  # fitting the model

levels(train_val$classe)[1] <- "0"   # conveting depend variable of validation dataset
levels(train_val$classe)[2] <- "1"
levels(train_val$classe)[3] <- "2"
levels(train_val$classe)[4] <- "3"
levels(train_val$classe)[5] <- "4"

train_val_y <- train_val$classe
train_tr$classe <- NULL

dmy <- dummyVars("~ .",data = train_val)   # converting each column to numeric type
train_val_dum <- data.frame(predict(dmy,newdata=train_val))
pred_xgb = predict(bst1,as.matrix(train_val_dum))  # predict
confusionMatrix(pred_xgb, train_val_y)

```



#Prediction on testing data
The accuracy is 99.91% of rf model which is the best that I have got till now, thus out-of-sample error is  100 - 99.91 = 0.09%
```{r}
final_pred <- predict(model_rf,testing) # final prediction on testing dataset
final_pred
```

# Submission file 

```{r}
output_file <- function(x){
  for (i in 1:length(x)){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,row.names = F,col.names = F,quote = F)
  }
}
output_file(final_pred)  # prediction file to submit
```
