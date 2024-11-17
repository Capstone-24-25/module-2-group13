library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(caret)
library(glmnet)
library(pROC)

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

text_bigram <- claims_clean %>% 
  unnest_tokens(output = token,
                input = text_clean,
                token = 'ngrams',
                n = 2,
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

# LPCR
set.seed(123)
split <- initial_split(text_word, 
                       prop = 0.8,
                       strata = 'bclass')

word_train <- training(split)
word_test <- testing(split)

y_train <- word_train$bclass

features_train <- word_train %>% select(-.id,-bclass, -mclass)
features_train <- features_train[, apply(features_train, 2, var) != 0]
features_test <- word_test %>% select(-.id,-bclass, -mclass)
features_test <- features_test[, apply(features_test, 2, var) != 0]

#train
pca_model <- prcomp(features_train, scale. = TRUE)
explained_variance <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
num_components <- which(explained_variance >= 0.9)[1]

# Transform both training and testing datasets using the same PCA model
pca_train <- predict(pca_model, newdata = word_train)
pca_test <- predict(pca_model, newdata = word_test)

logistic_model <- glm(y_train ~ ., data = as.data.frame(pca_train), family = "binomial")

predictions <- predict(logistic_model, newdata = as.data.frame(pca_test), type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
accuracy <- mean(predicted_classes == pca_test$bclass)

roc_curve <- roc(word_test$bclass, predictions)
auc(roc_curve)

