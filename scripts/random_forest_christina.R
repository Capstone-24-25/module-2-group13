library(tidyverse)
library(dplyr)
library(tidymodels)
library(readr)
library(glmnet)
library(randomForest)
library(tidytext)
library(keras3)
library(caret)
library(textrecipes)
library(recipes)

load("~/Desktop/UCSB/Statistics & Data Science/PSTAT 197/module-2-group13/data/claims-clean-example.RData")

set.seed(188)

partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1


# Convert labels to factor for classification
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.factor()  

test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.factor() 

text_recipe <- recipe(bclass ~ text_clean, data = training(partitions)) %>%
  step_tokenize(text_clean) %>%
  step_stopwords(text_clean) %>%
  step_tfidf(text_clean, min_times = 5)  # Keep tokens that appear in at least 5 documents

text_recipe <- recipe(bclass ~ text_clean, data = training(partitions)) %>%
  step_tokenize(text_clean) %>%                          # Tokenize text
  step_stopwords(text_clean) %>%                         # Remove stopwords
  step_tokenfilter(text_clean, max_tokens = 500) %>%     # Limit to 500 tokens
  step_tfidf(text_clean)                                 # Compute TF-IDF

prepared_recipe <- prep(text_recipe)

# Generate feature matrices for training and testing
train_features <- bake(prepared_recipe, new_data = NULL)     # Training data
test_features <- bake(prepared_recipe, new_data = testing(partitions))  # Test data

rf_model <- randomForest(
  x = select(train_features, -bclass),  # Exclude the target column
  y = train_features$bclass,           # Use the target column as response
  ntree = 500,                         # Number of trees
  mtry = sqrt(ncol(train_features) - 1),  # Number of predictors considered at each split
  importance = TRUE
)

# Evaluate performance on the test set
test_predictions <- predict(rf_model, newdata = select(test_features, -bclass))
binary_accuracy <- mean(test_predictions == test_features$bclass)
print(paste("Binary Accuracy:", round(binary_accuracy, 4)))

