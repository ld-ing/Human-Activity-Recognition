# Human-Activity-Recognition

### Project for DSC 450 (Data Science Practicum)
#### Group Member - K. Hu, L. Ding, T. Xie, J. Shang, K. Lyu

## Essential Infomation
The data set is originally from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/).  
The goal of our project is to 'Find a predictive model that uses as few features as possible'. We use Random Forest Algorithm to select some most important features and build XGBoost model on all the combinations of features. After selecting the best model by the result of validation set, we test our model on the test set and get a fairly good result.  

## About the Code
Tipically, we use R to do the whole thing.  
main.r is the main programming .r file. It contains all the parts (pre-processing, feature selection, modeling, and evaluation) of our project.  
samsungData.rda is the .rda file which contains data after some kind of cleaning, provided by the instructor. It is used by our r code. If you want to run the code, it would be better that you have this .rda file in the same directory as the r code. You can simply do this by cloning the whole repository.  
