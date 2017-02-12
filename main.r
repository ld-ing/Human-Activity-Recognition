# ----------
# DSC 450 Data Science Practicum
# Mini-Project: Human Activity Recognition 
# Li Ding
# Feb. 4th, 2017
# ----------



# ---------- Data Preparation -------------
# Uncomment the following line if you don't have rstudioapi installed
install.packages("rstudioapi")

# Set working directory the same as the script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load('samsungData.rda')  # Be sure this is also in the working directory

# Data pre-processing
samsungData <- samsungData[!duplicated(lapply(samsungData, summary))]
featureName <- colnames(samsungData)  # save original feature names after unique
colnames(samsungData) <- make.names(c(1:ncol(samsungData)))  # make clear column names, totally 542
samsungData$X542 <- as.factor(samsungData$X542)  # y as a factor
samsungData$X542 <- as.numeric(samsungData$X542)-1  # make y 0,1,2.. for XGBoost
samsungData <- samsungData[,-541]  # drop useless column
# Now we have 7352 obs. of 541 variables


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
model <- randomForest(X542~., data = train, ntree=100)

# The 20 most important features
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
  traini <- train[,c(541,f[flist[[i]]])]
  vali <- val[,c(541,f[flist[[i]]])]
  testi <- test[,c(541,f[flist[[i]]])]
  dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
  dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
  dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
  
  # Modeling
  model <- xgb.train(data = dtraini, max.depth = 6, nrounds = 1000,
                     objective = "multi:softmax", num_class = 6, verbose = 1, 
                     watchlist = list(validation = dvali), maximize = FALSE, 
                     early.stop.round = 30, print.every.n = 1000, salient = 1, eta = 0.1)
  
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
# Minimal Validation Error 0.1788171
result[which.min(result$val_error), -3]
# Best round 28, feature 539, 84 are selected
besti <- which.min(result$val_error)  # besti = 28


# Testing the best model with test data
i <- besti
traini <- train[,c(541,f[flist[[i]]])]
vali <- val[,c(541,f[flist[[i]]])]
testi <- test[,c(541,f[flist[[i]]])]
dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
model <- bestmodel
trainerr <- sum(predict(model,dtraini) != traini[,1])/nrow(traini)
valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
testerr <- sum(predict(model,dtesti) != testi[,1])/nrow(testi)

result <- cbind(trainerr, valerr, testerr)
colnames(result) <- c('training_error','validation_error','test_error')
rownames(result) <- 'feature 539, 84'

result
# training_error   validation_error   test_error
#  0.1315585         0.1788171         0.1724138
# The test accuracy is 1 - 0.1724138 = 0.8275862, already above 80%.





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
  traini <- train[,c(541,f[flist[[i]]])]
  vali <- val[,c(541,f[flist[[i]]])]
  testi <- test[,c(541,f[flist[[i]]])]
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
# 0.06327373
result[which.min(result$val_error),-4]
# round 279, feature 539, 53, 365
besti <- which.min(result$val_error)

# Testing the best model with test data
i <- besti
traini <- train[,c(541,f[flist[[i]]])]
vali <- val[,c(541,f[flist[[i]]])]
testi <- test[,c(541,f[flist[[i]]])]
dtraini <- xgb.DMatrix(data = data.matrix(traini[,-1]), label = traini[,1])
dvali <- xgb.DMatrix(data = data.matrix(vali[,-1]), label = vali[,1])
dtesti <- xgb.DMatrix(data = data.matrix(testi[,-1]), label = testi[,1])
model <- bestmodel
trainerr <- sum(predict(model,dtraini) != traini[,1])/nrow(traini)
valerr <- sum(predict(model,dvali) != vali[,1])/nrow(vali)
testerr <- sum(predict(model,dtesti) != testi[,1])/nrow(testi)

result <- cbind(trainerr, valerr, testerr)
colnames(result) <- c('training_error','validation_error','test_error')
rownames(result) <- 'feature 539, 53, 365'

result
# training_error   validation_error    test_error
#  0.004469491        0.06327373        0.0831643
# The test accuracy is 1 - 0.0831643 = 0.9168357, already above 90%.



