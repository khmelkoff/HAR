#####################################################################
# project: HAR
# script ver.: 2 (+pca)
# author: khmelkoff@gmail.com
# date: 01/05/2015
#####################################################################

# 1.Load packages ###################################################

library(FSelector) # need JRE!
library(caret)
library(randomForest)
library(kernlab)
library(e1071)

# 2.Load data #######################################################
## set up urls for datasets
### common data
url_variables <- "./UCI HAR Dataset/features.txt"
url_activity_names <- "./UCI HAR Dataset/activity_labels.txt"
### training data
url_train_data <- "./UCI HAR Dataset/train/X_train.txt"
url_train_activity <- "./UCI HAR Dataset/train/y_train.txt"
url_train_subjects <- "./UCI HAR Dataset/train/subject_train.txt"
### test data
url_test_data <- "./UCI HAR Dataset/test/X_test.txt"
url_test_activity <- "./UCI HAR Dataset/test/y_test.txt"
url_test_subjects <- "./UCI HAR Dataset/test/subject_test.txt"


## load common data
activity_names <- read.table(url_activity_names, stringsAsFactors=F)
var_names <- read.table(url_variables,  stringsAsFactors=F)

## load training data
train_data <- read.table(url_train_data)
train_activity <- read.table(url_train_activity)

## load test data
test_data <- read.table(url_test_data)
test_activity <- read.table(url_test_activity)

## Merges data ######################################################
## correct variable names

## editNames function for the substitution for variable names
editNames <- function(x) {
  y <- var_names[x,2]
  y <- sub("BodyBody", "Body", y) #subs duplicate names
  y <- gsub("-", "", y) # global subs for dash
  y <- gsub(",", "_", y) # global subs for comma
  y <- sub("\\()", "", y) # subs for ()
  y <- gsub("\\)", "", y) # global subs for 
  y <- sub("\\(", "_", y) # subs for (
  y <- paste0("v",var_names[x,1], "_",y) #add number, prevent duplicat.   
  return(y)
}
## edit names
new_names <- sapply(1:nrow(var_names), editNames)

## work with training data
names(train_data)<-new_names
train_data <- cbind(train_activity[,1], train_data)
names(train_data)[1]<-"Activity"

## work with test data
names(test_data)<-new_names
test_data <- cbind(test_activity[,1], test_data)
names(test_data)[1]<-"Activity"

activity_names[2,2] <- substr(activity_names[2,2], 1, 10) #cut long names
activity_names[3,2] <- substr(activity_names[3,2], 1, 12)

train_data <- transform(train_data, Activity=factor(Activity))
test_data <- transform(test_data, Activity=factor(Activity))
levels(train_data[,1])<-activity_names[,2]
levels(test_data[,1])<-activity_names[,2]


# 3. Data exploration ###############################################
## check range
rng <- sapply(new_names, function(x){
    range(train_data[,x])  
})
min(rng)
max(rng)

## check skewness
SkewValues <- apply(train_data[,-1], 2, skewness)
head(SkewValues[order(abs(SkewValues),decreasing = T)],3)

## activity distribution
summary(train_data$Activity)

# 4. Full models ####################################################
## Random Forest
fitControl <- trainControl(method="cv", number=5)
set.seed(123)
tstart <- Sys.time()
forest_full <- train(Activity~., data=train_data,
                        method="rf", do.trace=10, ntree=100,
                        trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(forest_full, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## SVM, full set
fitControl <- trainControl(method="cv", number=5)
tstart <- Sys.time()
svm_full <- train(Activity~., data=train_data,
                     method="svmRadial",
                     trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(svm_full, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

# 5. Model with important variables #################################

plot(varImp(forest_full),20, scales=list(cex=1.1))

## % variable extraction ######################
imp <- varImp(forest_full)[[1]]
imp_vars <- rownames(imp)[order(imp$Overall, decreasing=TRUE)]
vars <- imp_vars[1:490] # % features

## model
fitControl <- trainControl(method="cv", number=5)
tstart <- Sys.time()
svm_imp <- train(Activity~., data=train_data[,c("Activity", vars)],
                 method="svmRadial",
                 trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

prediction <- predict(svm_imp, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

# 6. Information gain ###############################################
## calculate ratio
inf_g <- information.gain(Activity~., train_data)
inf_gain <- cbind.data.frame(new_names, inf_g, stringsAsFactors=F)
names(inf_gain) <- c("vars", "ratio")
row.names(inf_gain) <- NULL
## arrange by ratio descending and plot top-20 variables
inf_gain <- inf_gain[order(inf_gain$ratio, decreasing=TRUE),]
dotplot(factor(vars, levels=rev(inf_gain[1:20,1])) ~ ratio, 
        data=inf_gain[1:20,],
        scales=list(cex=1.1))

inf_gain[10,1]
## [1] "tBodyAccmadX"
plot(train_data[,inf_gain[10,1]], ylab=inf_gain[10,1],
     pch=20, col=train_data[,1], main="IGR = 0.87")
legend("topright", pch=20, col=activity_names[,1],
       legend=activity_names[,2], cex=0.8)

inf_gain[551,1]
## [1] "tBodyAccJerkMagarCoeff4"
plot(train_data[,inf_gain[551,1]], ylab=inf_gain[551,1],
     pch=20, col=train_data[,1], main="IGR = 0.03")
legend("topright", pch=20, col=activity_names[,1],
       legend=activity_names[,2], cex=0.8)

## select variables (igr cutoff) ################ 
vars <- inf_gain$vars[1:547]

## SVM with best igr variables ##################
## for parallel processing
# library(doMC) # don't use for Windows
# registerDoMC(cores=3) # don't use for Windows

fitControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
tstart <- Sys.time()
svm_igr <- train(Activity~., data=train_data[,c("Activity", vars)],
                 method="svmRadial",
                 trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

prediction <- predict(svm_igr, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## Random Forest ################################

vars <- inf_gain$vars[1:526] # Accuracy = 0.9243
fitControl <- trainControl(method="cv", number=5)
set.seed(123)
tstart <- Sys.time()
forest_igr <- train(Activity~., data=train_data[,c("Activity", vars)],
                    method="rf", do.trace=10, ntree=100,
                    trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(forest_igr, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## PCA ##############################################################

pca_mod <- preProcess(train_data[,-1],
                      method="pca",
                      thresh = 0.95)

summary(pca_mod)

pca_train_data <- predict(pca_mod, newdata=train_data[,-1])
dim(pca_train_data)
# [1] 7352  102
pca_train_data$Activity <- train_data$Activity
pca_test_data <- predict(pca_mod, newdata=test_data[,-1])
pca_test_data$Activity <- test_data$Activity

## RF with pca data #######################################
fitControl <- trainControl(method="cv", number=5)
set.seed(123)
tstart <- Sys.time()
forest_pca <- train(Activity~., data=pca_train_data,
                    method="rf", do.trace=10, ntree=100,
                    trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(forest_pca, newdata=pca_test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

# Accuracy : 0.8734

## SVM with pca data ######################################
fitControl <- trainControl(method="cv", number=5)
set.seed(123)
tstart <- Sys.time()
svm_pca <- train(Activity~., data=pca_train_data,
                    method="svmRadial",
                    trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(svm_pca, newdata=pca_test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

# Accuracy : 0.9386


