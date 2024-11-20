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

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

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
partitions_svmb <- text_word %>% initial_split(text_word, 
                                          prop = 0.7,
                                          strata = 'bclass')

test_dtm_svmb <- testing(partitions_svmb) %>%
  select(-.id, -bclass, -mclass)
test_labels_svmb <- testing(partitions_svmb) %>%
  select(.id, bclass, mclass)

train_dtm_svmb <- training(partitions_svmb) %>%
  select(-.id, -bclass, -mclass)
train_labels_svmb <- training(partitions_svmb) %>%
  select(.id, bclass, mclass)

proj_out_svmb <- projection_fn(.dtm = train_dtm_svmb, .prop = 0.7)
train_dtm_projected_svmb <- proj_out_svmb$data

test_proj_svmb <- reproject_fn(.dtm = test_dtm_svmb, 
                          proj_out_svmb)

train_svmb <- train_labels_svmb %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected_svmb)

svm_model <- svm(bclass ~ ., data = train_svmb, kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions <- predict(svm_model, newdata = as.matrix(test_proj_svmb))
confusion_matrix_svmb <- table(Predicted = svm_predictions, Actual = test_labels_svmb$bclass)
accuracy_svmb <- sum(diag(confusion_matrix_svmb)) / sum(confusion_matrix_svmb) #0.8074324

############

####### mclass
set.seed(123)
partitions_svm_m <- text_word %>% initial_split(text_word, 
                                               prop = 0.7,
                                               strata = 'mclass')

test_dtm_svm_m <- testing(partitions_svm_m) %>%
  select(-.id, -bclass, -mclass)
test_labels_svm_m <- testing(partitions_svm_m) %>%
  select(.id, bclass, mclass)

train_dtm_svm_m <- training(partitions_svm_m) %>%
  select(-.id, -bclass, -mclass)
train_labels_svmb_m <- training(partitions_svm_m) %>%
  select(.id, bclass, mclass)

proj_out_svm_m <- projection_fn(.dtm = train_dtm_svm_m, .prop = 0.7)
train_dtm_projected_svm_m <- proj_out_svm_m$data

test_proj_svm_m <- reproject_fn(.dtm = test_dtm_svm_m, 
                               proj_out_svm_m)

train_svm_m <- train_labels_svmb_m %>%
  transmute(mclass = factor(mclass)) %>%
  bind_cols(train_dtm_projected_svm_m)

svm_model_m <- svm(mclass ~ ., data = train_svm_m, kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions_m <- predict(svm_model_m, newdata = as.matrix(test_proj_svm_m))

confusion_matrix_svm_m <- table(Predicted = svm_predictions_m, Actual = test_labels_svm_m$mclass)
accuracy_svm_m <- sum(diag(confusion_matrix_svm_m)) / sum(confusion_matrix_svm_m) #0.7790894
