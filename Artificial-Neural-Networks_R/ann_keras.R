library(keras)
library(tidyverse)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
set.seed(333)

# Importing the dataset
dataset <- read.csv('Churn_Modelling.csv')%>%
  select(c(4:14))

#Split into train and test
train_test_split <- initial_split(dataset, prop = 0.8)

# Retrieve train and test sets
train <- training(train_test_split)
test  <- testing(train_test_split)  
  
# scale and factor      
rec_obj <- recipe(Exited ~ ., data = train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train)


# Predictors
x_train <- bake(rec_obj, newdata = train)%>%
                select(-Exited)
x_test  <- bake(rec_obj, newdata = test)%>%
                select(-Exited)

# Response variables for training and testing sets
y_train <- pull(train, Exited)
  
y_test  <- pull(test, Exited)


# Building our Artificial Neural Network
 
# initalize sequence for ANN
model<-keras_model_sequential()%>%
  # First hidden layer
  layer_dense(units = 16, kernel_initializer = "uniform", activation = "relu", input_shape = ncol(x_train)) %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(units = 16, kernel_initializer = "uniform", activation = "relu") %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid") %>% 
  # Compile ANN
  compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = c('accuracy'))


# Fit the keras model to the training data
fit <- fit(object = model, x = as.matrix(x_train), y = y_train, batch_size = 50, epochs = 15, validation_split = 0.30)


#Predicted Class
yhat_keras_class_vec <- predict_classes(object = model, x = as.matrix(x_test)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model, x = as.matrix(x_test)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)

