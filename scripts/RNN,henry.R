library(tidyverse)
library(keras3)
library(caret)
library(tensorflow)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(sparsesvd)
library(Matrix)
library(glmnet)
library(tidymodels)

#using claims_clean as training
load("~/Downloads/module-2-group13/data/claims-clean-example.RData")

#using train_tf_idf as training set
train_dtm <- claims_clean %>% 
  unnest_tokens(output = token, 
                input = text_clean,
                token = 'words',
                stopwords = stpwrd) %>% # optional stopword removal
  mutate(token = lemmatize_words(token)) %>% 
  filter(str_length(token) > 2) %>% 
  count(.id, mclass, bclass, token, name = 'n') %>%
  bind_tf_idf(term = token,
              document = mclass,
              n = n) %>% 
  pivot_wider(id_cols = c(.id, mclass, bclass),
              names_from = token,
              values_from = tf_idf,
              values_fill = 0)

x_train <- train_dtm %>%
  ungroup() %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

y_train <- train_dtm %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1

#NN
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>% 
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

model %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'binary_accuracy'
  )

history <- model %>%
  fit(x = x_train,
      y = y_train,
      epochs = 20)

#Using logistical NN we have binary_accuracy of 0.5475

rnn_model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(rnn_model)

rnn_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2)

#using RNN we found a binary accuracy of 0.8198 on the training set

#testing data
x_test <- test_tfidf %>%
  select(-.id) %>%
  as.matrix()

#predicting using rnn_model
preds <- predict(rnn_model, x_test,)

pred_df <- as.data.frame(preds) %>% 
  mutate(bclass.pred = factor(preds > 0.5, 
                              levels = c(T,F),
                              labels = c('NA','Relevant')))

#multiple classification
xm_train <- train_dtm %>%
  ungroup() %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

ym_train <- train_dtm %>% 
  pull(mclass) %>%
  factor()


multi_rnn_model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 5, activation = "softmax")

multi_rnn_model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",  # Multiclass loss
  metrics = c('accuracy')
)

summary(multi_rnn_model)

multi_rnn_model <- model %>% fit(
  xm_train,
  ym_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

#accuracy of 0.1098