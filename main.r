# ----------
# DSC 450 Data Science Practicum
# Mini-Project: Human Activity Recognition 
# Li Ding
# Feb. 4th, 2017
# ----------



# ---------- Data Preparation -------------
# Uncomment the following line if you don't have rstudioapi installed
#install.packages("rstudioapi")

# Set working directory the same as the script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load('samsungData.rda')  # Be sure this is also in the working directory

# Data pre-processing
colnames(samsungData) <- make.names(c(1:563))  # make clear column names
samsungData$X563 <- as.factor(samsungData$X563)  # y as a factor
samsungData$X563 <- as.numeric(samsungData$X563)-1  # make y 0,1,2.. for XGBoost
samsungData <- samsungData[,-562]  # drop useless column

# Briefly, we do a 7:1:2 split for training, validation, and test.
set.seed(822)
id <- sample(1:nrow(samsungData),0.7*nrow(samsungData))
train <- samsungData[id,]
test <- samsungData[-id,]
id <- sample(1:nrow(test),0.33*nrow(test))
val <- test[id,]
test <- test[-id,]




# ---------- Feature Selection -------------
library(randomForest)

set.seed(822)
model <- randomForest(X563~., data = train, ntree=100)

# The 10 most important features
f <- tail(order(model$importance),20)





# ---------- Modeling -------------
library(xgboost)

# 2-feature case
set.seed(822)

# Create a list of all the combinations of 2 features in 20 features
flist = combn(1:20, 2, simplify = FALSE)  

# Initialize some parameters
result = data.frame()
bestmodel <- NULL
bestvalerr <- 1

# Build models for all possible feature combinations
for (i in 1:choose(20,2)){
  cat("Starting with round: ")
  cat(i)
  cat('\n')
  
  # Subsample the data with only selected features
  traini <- train[,c(562,f[flist[[i]]])]
  vali <- val[,c(562,f[flist[[i]]])]
  testi <- test[,c(562,f[flist[[i]]])]
  dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
  dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
  dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
  
  # Modeling
  model <- xgb.train(data = dtraini, max.depth = 6, nrounds = 1000, objective = "multi:softmax", num_class = 6, verbose = 1, watchlist = list(validation = dvali), maximize = FALSE, early.stop.round = 30, print.every.n = 1000, salient = 1, eta = 0.1)
  
  # Save the best model
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
  
  # Save the running results
  result <- rbind(result,c(f[flist[[i]]],valerr))
}

colnames(result) <- c('feature_selected_1', 'feature_selected_2', 'val_error')

min(result$val_error)
# Minimal Validation Error 0.1815681
result[which.min(result$val_error),-3]
# Best round 9, feature 42, 84 are selected
besti <- which.min(result$val_error)  # 9


# Testing the best model with test data
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
#  0.1352507         0.1815681        0.1757945
# The test accuracy is 1 - 0.1757945 = 0.8242055, already above 80%.





# 3-feature case
set.seed(822)

# Create a list of all the combinations of 3 features in 20 features
flist = combn(1:20, 3, simplify = FALSE)

# Initialize some parameters
result = data.frame()
bestmodel <- NULL
bestvalerr <- 1

# Build models for all possible feature combinations
for (i in 1:choose(20,3)){
  cat("Starting with round: ")
  cat(i)
  cat('\n')
  
  # Subsample the data with only selected features
  traini <- train[,c(562,f[flist[[i]]])]
  vali <- val[,c(562,f[flist[[i]]])]
  testi <- test[,c(562,f[flist[[i]]])]
  dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
  dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
  dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
  
  # Modeling
  model <- xgb.train(data = dtraini, max.depth = 6, nrounds = 1000, objective = "multi:softmax", num_class = 6, verbose = 1, watchlist = list(validation = dvali), maximize = FALSE, early.stop.round = 30, print.every.n = 1000, salient = 1, eta = 0.1)
  
  # Save the best model
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
  
  # Save the running results
  result <- rbind(result,c(f[flist[[i]]],valerr))
}

colnames(result) <- c('feature_selected_1', 'feature_selected_2', 'feature_selected_3', 'val_error')

min(result$val_error)
# 0.0866575
result[which.min(result$val_error),-4]
# round 130, feature 42, 53, 394
besti <- which.min(result$val_error)

# Testing the best model with test data
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
#  0.01146522        0.0866575        0.0831643
# The test accuracy is 1 - 0.0831643 = 0.9168357, already above 90%.



