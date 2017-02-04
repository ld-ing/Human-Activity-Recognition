# ----------
# DSC 450 Data Science Practicum
# Mini-Project: Human Activity Recognition 
# Li Ding
# Jan. 27, 2017
# ----------



# ---------- Data Preparation -------------
# uncomment the following line if you don't have rstudioapi installed
#install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load('samsungData.rda')  # Be sure this is also under the working directory

colnames(samsungData) <- make.names(c(1:563))
samsungData$X563 <- as.factor(samsungData$X563)
samsungData$X563 <- as.numeric(samsungData$X563)-1
samsungData <- samsungData[,-562]

set.seed(822)
id <- sample(1:nrow(samsungData),0.7*nrow(samsungData))
train <- samsungData[id,]
test <- samsungData[-id,]
id <- sample(1:nrow(test),0.5*nrow(test))
val <- test[id,]
test <- test[-id,]




# ---------- Feature Selection -------------
library(randomForest)

set.seed(822)
model <- randomForest(X563~., data = train, ntree=100)
# All Features Test Error
#sum(test$X563 != predict(model,test))/nrow(test)
# The 10 most important features
f <- tail(order(model$importance),20)




# ---------- Modeling -------------
library(xgboost)
library(foreach)


# 2-feature case
set.seed(822)
flist = combn(1:20, 2, simplify = FALSE)
result = data.frame(feature_selected_1 = integer(),feature_selected_2 = integer(), val_error = double())
bestmodel <- NULL
bestvalerr <- 1

for (i in 1:choose(20,2)){
  cat("Starting with round: ")
  cat(i)
  cat('\n')
  traini <- train[,c(562,f[flist[[i]]])]
  vali <- val[,c(562,f[flist[[i]]])]
  testi <- test[,c(562,f[flist[[i]]])]
  dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
  dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
  dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
  model <- xgb.train(data = dtraini, max.depth = 6, nrounds = 1000, objective = "multi:softmax", num_class = 6, verbose = 1, watchlist = list(validation = dvali), maximize = FALSE, early.stop.round = 30, print.every.n = 1000, salient = 1, eta = 0.1)
  valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
  if (valerr < bestvalerr){
    bestvalerr = valerr
    bestmodel = model
  }
  cat("\nFeature Selected: ")
  cat(f[flist[[i]]])
  cat("\nVal Error: ")
  cat(valerr)
  cat('\n')
  cat('\n')
  result <- rbind(result,c(f[flist[[i]]],valerr))
}


colnames(result) <- c('feature_selected_1', 'feature_selected_2', 'val_error')

min(result$val_error)
# 0.1795104
result[which.min(result$val_error),-3]
# round 9, feature 42,84
besti <- which.min(result$val_error)

i <- besti
traini <- train[,c(562,f[flist[[i]]])]
vali <- val[,c(562,f[flist[[i]]])]
testi <- test[,c(562,f[flist[[i]]])]
dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
model <- bestmodel
trainerr <- sum(predict(model,dtraini) != traini[,1])/nrow(traini)
valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
testerr <- sum(predict(model,dtesti) != testi[,1])/nrow(testi)

result <- cbind(trainerr, valerr, testerr)
colnames(result) <- c('training_error','validation_error','test_error')
rownames(result) <- 'feature 42, 84'

result
# training_error   validation_error   test_error
#  0.1352507         0.1795104         0.175884





# 3-feature case
set.seed(822)
flist = combn(1:20, 3, simplify = FALSE)
result = data.frame(feature_selected_1 = integer(),feature_selected_2 = integer(), feature_selected_3 = integer(), val_error = double())
bestmodel <- NULL
bestvalerr <- 1

for (i in 1:choose(20,3)){
  cat("Starting with round: ")
  cat(i)
  cat('\n')
  traini <- train[,c(562,f[flist[[i]]])]
  vali <- val[,c(562,f[flist[[i]]])]
  testi <- test[,c(562,f[flist[[i]]])]
  dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
  dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
  dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
  model <- xgb.train(data = dtraini, max.depth = 6, nrounds = 1000, objective = "multi:softmax", num_class = 6, verbose = 1, watchlist = list(validation = dvali), maximize = FALSE, early.stop.round = 30, print.every.n = 1000, salient = 1, eta = 0.1)
  valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
  if (valerr < bestvalerr){
    bestvalerr = valerr
    bestmodel = model
  }
  cat("\nFeature Selected: ")
  cat(f[flist[[i]]])
  cat("\nVal Error: ")
  cat(valerr)
  cat('\n')
  cat('\n')
  result <- rbind(result,c(f[flist[[i]]],valerr))
}


colnames(result) <- c('feature_selected_1', 'feature_selected_2', 'feature_selected_3', 'val_error')

min(result$val_error)
# 0.0843155
result[which.min(result$val_error),-4]
# round 130, feature 42, 53, 394
besti <- which.min(result$val_error)

i <- besti
traini <- train[,c(562,f[flist[[i]]])]
vali <- val[,c(562,f[flist[[i]]])]
testi <- test[,c(562,f[flist[[i]]])]
dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
model <- bestmodel
trainerr <- sum(predict(model,dtraini) != traini[,1])/nrow(traini)
valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
testerr <- sum(predict(model,dtesti) != testi[,1])/nrow(testi)

result <- cbind(trainerr, valerr, testerr)
colnames(result) <- c('training_error','validation_error','test_error')
rownames(result) <- 'feature 42, 53, 394'

result
# training_error   validation_error   test_error
#  0.008356005        0.0843155       0.08522212




