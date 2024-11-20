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


#########svm bclass
set.seed(123)
split <- initial_split(text_word, 
                       prop = 0.7,
                       strata = 'bclass')

svm_bclass_train <- training(split) %>% select(-.id, -mclass)
svm_bclass_test <- testing(split) %>% select(-.id, -mclass)

svm_y_train <- svm_bclass_train$bclass

features_train_svm <- svm_bclass_train %>% select(-bclass)
features_train_svm <- features_train_svm[, apply(features_train_svm, 2, var) != 0]
features_test_svm <- svm_bclass_test %>% select(-bclass)
features_test_svm <- features_test_svm[, apply(features_test_svm, 2, var) != 0]

#train model
pca_model_svm <- prcomp(features_train_svm, scale. = TRUE)

# Transform both training and testing datasets using the same PCA model
pca_train_svm <- predict(pca_model_svm, newdata = svm_bclass_train)
pca_test_svm <- predict(pca_model_svm, newdata = svm_bclass_test)

svm_model <- svm(svm_y_train ~ ., data = pca_train_svm, kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions <- predict(svm_model, newdata = pca_test_svm)

#svm_predicted_classes <- ifelse(svm_predictions > 0.5, 1, 0)

roc_curve_b <- roc(svm_bclass_test$bclass, svm_predictions)
auc(roc_curve_b)
##Area under the curve: 0.4571

####### mclass
set.seed(123)
split_mclass <- initial_split(text_word, 
                       prop = 0.7,
                       strata = 'mclass')

svm_mclass_train <- training(split_mclass) %>% select(-.id, -bclass)
svm_mclass_test <- testing(split_mclass) %>% select(-.id, -bclass)

svm_y_train_mclass <- svm_mclass_train$mclass

features_train_svm_m <- svm_mclass_train %>% select(-mclass)
features_train_svm_m <- features_train_svm_m[, apply(features_train_svm_m, 2, var) != 0]
features_test_svm_m <- svm_mclass_test %>% select(-mclass)
features_test_svm_m <- features_test_svm_m[, apply(features_test_svm_m, 2, var) != 0]

#train model
pca_model_m <- prcomp(features_train_svm_m, scale. = TRUE)

# Transform both training and testing datasets using the same PCA model
pca_train_svm_m <- predict(pca_model_m, newdata = svm_mclass_train)
pca_test_svm_m <- predict(pca_model_m, newdata = svm_mclass_test)

svm_model_m <- svm(svm_y_train_mclass ~ ., data = pca_train_svm_m, kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions_m <- predict(svm_model_m, newdata = pca_test_svm_m)

##svm_predicted_classes_m <- ifelse(svm_predictions_m > 0.5, 1, 0)

roc_curve_m <- roc(svm_mclass_test$mclass, svm_predictions_m)
auc(roc_curve_m)

confusion_matrix <- table(Predicted = svm_predictions_m, Actual = svm_mclass_test$mclass)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy
## 0.5042159