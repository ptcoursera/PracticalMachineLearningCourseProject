rm(list = ls())

library(caret)
library(kernlab)
set.seed(1000)

data.raw <- read.csv(file="pml-training.csv", header=TRUE, sep=",")

smallvarinds = nearZeroVar(data.raw)
nainds = which(vapply(data.raw, function(x){sum(is.na(x))>0}, TRUE))
userselectedinds = 1:7 #indices with training set specific information needs to be be deleted

removedindices = unique(c(nainds, smallvarinds, userselectedinds))
data.raw <- data.raw[,-removedindices]

inTrain <- createDataPartition(y=data.raw$classe, p=0.6,list=FALSE)

training <- data.raw[inTrain,]
testing <- data.raw[-inTrain,]

#this following line outputs a cost value of 10000.
# svmhp <- tune.svm(classe~., data = training, cost = 10^(0:4)) 

## train control options
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1,
                           )

## train classifiers
model.svm <- ksvm(classe~.,data=training,kernel="rbfdot",C=10000, cross=10)

gbmGrid <-  expand.grid(interaction.depth = 5,n.trees = 200,shrinkage = 0.1)
model.gbm <- train(classe ~ .,data=training, method="gbm", tuneGrid=gbmGrid, trControl=fitControl)

knnGrid <- expand.grid(k = c(1)) 
model.knn <- train(classe ~ ., data=training, method="knn", tuneGrid = knnGrid, trControl=fitControl)


## predict after training
preds.gbm = predict(model.gbm,newdata=testing)
preds.svm = predict(model.svm,newdata=testing)
preds.knn = predict(model.knn, newdata=testing)

actual = testing$classe

ba = data.frame()
cf.gbm = confusionMatrix(preds.gbm, actual)
ba = rbind(ba, data.frame(type = rep('GBM',5), class = c("A", "B", "C","D","E"),acc = cf.gbm$byClass[,8]))
cf.svm = confusionMatrix(preds.svm, actual)
ba = rbind(ba, data.frame(type = rep('SVM',5), class = c("A", "B", "C","D","E"),acc = cf.svm$byClass[,8]))
cf.knn = confusionMatrix(preds.knn, actual)
ba = rbind(ba, data.frame(type = rep('KNN',5), class = c("A", "B", "C","D","E"),acc = cf.knn$byClass[,8]))

library(lattice)
barchart(acc~type,data=ba,groups=class, main="Balanced Accuracies for different classes grouped by classifier", ylim=c(0.7, 1), auto.key=TRUE, ylab="")

#perform pca

end = ncol(training)
preObj <- preProcess(training[,-end],method="pca", pcaComp=5)
pcatraining <- predict(preObj, training[,-end])
pcatraining$classe <-training$classe

pca.model.svm <- ksvm(classe~.,data=pcatraining,kernel="rbfdot",C=10000, cross=10)
pca.model.gbm <- train(classe ~ .,data=pcatraining, method="gbm", tuneGrid=gbmGrid,trControl=fitControl)
pca.model.knn <- train(classe ~ ., data=pcatraining, method="knn", tuneGrid = knnGrid,trControl=fitControl)

pcatesting = predict(preObj, testing[,-end])
pca.preds.gbm = predict(pca.model.gbm, newdata=pcatesting)
pca.preds.svm = predict(pca.model.svm, newdata=pcatesting)
pca.preds.knn = predict(pca.model.knn, newdata=pcatesting)

pca.ba = data.frame()
pca.cf.gbm = confusionMatrix(pca.preds.gbm, actual)
pca.ba = rbind(pca.ba, data.frame(type = rep('GBM',5), class = c("A", "B", "C","D","E"),acc = pca.cf.gbm$byClass[,8]))
pca.cf.svm = confusionMatrix(pca.preds.svm, actual)
pca.ba = rbind(pca.ba, data.frame(type = rep('SVM',5), class = c("A", "B", "C","D","E"),acc = pca.cf.svm$byClass[,8]))
pca.cf.knn = confusionMatrix(pca.preds.knn, actual)
pca.ba = rbind(pca.ba, data.frame(type = rep('KNN',5), class = c("A", "B", "C","D","E"),acc = pca.cf.knn$byClass[,8]))

library(lattice)
barchart(acc~type,data=pca.ba,groups=class, main="Balanced Accuracies for different classes grouped by classifier w/ PCA", ylim=c(0.7, 1), auto.key=TRUE, ylab="")


#perform same steps on out of sample data

realtest = read.csv(file="pml-testing.csv", header=TRUE, sep=",")
realtest <- realtest[,-removedindices]