library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(caret)
library(glmnet)
library(pROC)
library(caret)
library(e1071)

load("./data/claims-clean-example.RData")

text_word <- claims_clean %>% 
  unnest_tokens(output = token,
                input = text_clean,
                token = 'words',
                stopwords = str_remove_all(stop_words$word, 
                                           '[[:punct:]]')) %>% 
  mutate(token.lem = lemmatize_words(token)) %>% 
  filter(str_length(token.lem) > 2) %>% 
  count(.id, bclass, mclass, token.lem, name = 'n') %>% 
  bind_tf_idf(term = token.lem,
              document = .id,
              n = n) %>% 
  pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)

set.seed(123)
split <- initial_split(text_word, 
                       prop = 0.7,
                       strata = 'bclass')

svm_bclass_train <- training(split) %>% select(-.id, -mclass)
svm_bclass_test <- testing(split) %>% select(-.id, -mclass)

svm_y_train <- svm_bclass_train$bclass

features_train_svm <- svm_bclass_train %>% select(-bclass)
features_train_svm <- features_train[, apply(features_train, 2, var) != 0]
features_test_svm <- svm_bclass_test %>% select(-bclass)
features_test_svm <- features_test[, apply(features_test, 2, var) != 0]

#train model
pca_model <- prcomp(features_train_svm, scale. = TRUE)

# Transform both training and testing datasets using the same PCA model
pca_train_svm <- predict(pca_model, newdata = svm_bclass_train)
pca_test_svm <- predict(pca_model, newdata = svm_bclass_test)

svm_model <- svm(svm_y_train ~ ., data = pca_train_svm, kernel = "linear", cost = 1)

svm_predictions <- predict(svm_model, newdata = pca_test_svm)

svm_predicted_classes <- ifelse(predictions > 0.5, 1, 0)

roc_curve <- roc(svm_bclass_test$bclass, predictions)
auc(roc_curve)
