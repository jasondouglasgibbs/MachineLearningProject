---
title: "Practical Machine Learning - Final Project"
author: "Jason Gibbs"
date: "5/17/2021"
output: 
        html_document:
         keep_md: yes
---



# Overview

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.


# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: (http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


# Loading Data and Libraries
Loading all the libraries and the data

```r
library(lattice)
library(ggplot2)
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.0.5
```

```r
library(kernlab)
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 4.0.5
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 4.0.5
```

```r
set.seed(42)
```


```r
traincsv <- read.csv("./data/pml-training.csv")
testcsv <- read.csv("./data/pml-testing.csv")

dim(traincsv)
```

```
## [1] 19622   160
```

```r
dim(testcsv)
```

```
## [1]  20 160
```

We see that there are 160 variables and 19622 observations in the training set and 20 for the test set.


# Cleaning the Data

Removing unnecessary variables. Starting with N/A variables.

```r
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .9] #removes highly NA columns
traincsv <- traincsv[,-c(1:7)] #removing other non-useful data
```

Removing near zero variance variables.

```r
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)
```

```
## [1] 19622    53
```

We slit the training set into a validation and training set. The testing set "testcsv" will be left alone, and used for the final quiz test cases. 

```r
inTrain <- createDataPartition(y=traincsv$classe, p=0.7, list=F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]
```


# Creating and Testing the Models
Models from the instruction were used, including: Decision Trees, Random Forest, Gradient Boosted Trees, and SVM. 
Set up control for training to use 3-fold cross validation. 

```r
control <- trainControl(method="cv", number=3, verboseIter=F)
```

## Decision Tree

**Model:** 


```r
mod_trees <- train(classe~., data=train, method="rpart", trControl = control, tuneLength = 5)
fancyRpartPlot(mod_trees$finalModel)
```

![](report_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

**Prediction:**


```r
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1520  489  478  429  150
##          B   24  353   30   14  122
##          C  105  221  411  119  235
##          D   25   76  107  402   82
##          E    0    0    0    0  493
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5402         
##                  95% CI : (0.5274, 0.553)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4005         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9080  0.30992  0.40058  0.41701  0.45564
## Specificity            0.6329  0.95997  0.86005  0.94107  1.00000
## Pos Pred Value         0.4958  0.65009  0.37672  0.58092  1.00000
## Neg Pred Value         0.9454  0.85286  0.87171  0.89178  0.89076
## Prevalence             0.2845  0.19354  0.17434  0.16381  0.18386
## Detection Rate         0.2583  0.05998  0.06984  0.06831  0.08377
## Detection Prevalence   0.5210  0.09227  0.18539  0.11759  0.08377
## Balanced Accuracy      0.7704  0.63494  0.63032  0.67904  0.72782
```

## Random Forest


```r
mod_rf <- train(classe~., data=train, method="rf", trControl = control, tuneLength = 5)

pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    0    0    0    0
##          B    2 1136    4    0    0
##          C    1    3 1022    7    0
##          D    0    0    0  956    5
##          E    0    0    0    1 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9961          
##                  95% CI : (0.9941, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9951          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9974   0.9961   0.9917   0.9954
## Specificity            1.0000   0.9987   0.9977   0.9990   0.9998
## Pos Pred Value         1.0000   0.9947   0.9894   0.9948   0.9991
## Neg Pred Value         0.9993   0.9994   0.9992   0.9984   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1930   0.1737   0.1624   0.1830
## Detection Prevalence   0.2839   0.1941   0.1755   0.1633   0.1832
## Balanced Accuracy      0.9991   0.9981   0.9969   0.9953   0.9976
```

## Gradient Boosted Trees


```r
mod_gbm <- train(classe~., data=train, method="gbm", trControl = control, tuneLength = 5, verbose = F)

pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1127    6    0    1
##          C    0    9 1013    8    4
##          D    0    0    6  952    5
##          E    0    0    1    4 1072
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9894, 0.9941)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9899          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9873   0.9876   0.9908
## Specificity            0.9993   0.9985   0.9957   0.9978   0.9990
## Pos Pred Value         0.9982   0.9938   0.9797   0.9886   0.9954
## Neg Pred Value         1.0000   0.9975   0.9973   0.9976   0.9979
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1915   0.1721   0.1618   0.1822
## Detection Prevalence   0.2850   0.1927   0.1757   0.1636   0.1830
## Balanced Accuracy      0.9996   0.9940   0.9915   0.9927   0.9949
```

## Support Vector Machine


```r
mod_svm <- train(classe~., data=train, method="svmLinear", trControl = control, tuneLength = 5, verbose = F)

pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1556  164   94   78   52
##          B   28  801   96   42  142
##          C   40   65  785   92   63
##          D   43   28   30  702   66
##          E    7   81   21   50  759
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7822          
##                  95% CI : (0.7714, 0.7926)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7228          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9295   0.7032   0.7651   0.7282   0.7015
## Specificity            0.9079   0.9351   0.9465   0.9661   0.9669
## Pos Pred Value         0.8004   0.7223   0.7512   0.8078   0.8268
## Neg Pred Value         0.9701   0.9292   0.9502   0.9478   0.9350
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2644   0.1361   0.1334   0.1193   0.1290
## Detection Prevalence   0.3303   0.1884   0.1776   0.1477   0.1560
## Balanced Accuracy      0.9187   0.8192   0.8558   0.8471   0.8342
```

## Results (Accuracy & Out of Sample Error)


```
##      accuracy oos_error
## Tree    0.540     0.460
## RF      0.996     0.004
## GBM     0.992     0.008
## SVM     0.782     0.218
```

**The best model is the Random Forest model, with 0.9960918 accuracy and 0.0039082 out of sample error rate. We find that to be a sufficient enough model to use for our test sets. ** 


# Predictions on Test Set

Running our test set to predict the classe (5 levels) outcome for 20 cases with the **Random Forest** model.

```r
pred <- predict(mod_rf, testcsv)
print(pred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


# Appendix

correlation matrix of variables in training set

```r
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot, method="color")
```

![](report_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

Plotting the models

```r
plot(mod_trees)
```

![](report_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

```r
plot(mod_rf)
```

![](report_files/figure-html/unnamed-chunk-14-2.png)<!-- -->

```r
plot(mod_gbm)
```

![](report_files/figure-html/unnamed-chunk-14-3.png)<!-- -->
