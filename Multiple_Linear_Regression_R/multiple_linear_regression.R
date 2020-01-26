# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = unique(dataset$State),
                       labels = c(1:length(unique(dataset$State))))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)



mse <- function(sm) 
  mean(sm$residuals^2)

library(caret)
folds = createFolds(training_set$Profit, k = 5)

cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  regressor = lm(formula = Profit ~ .,
                   data = training_fold)
  M <- mean((test_fold$Profit - predict(regressor, test_fold[-5])) ^ 2)
  return(M)
})
cv

