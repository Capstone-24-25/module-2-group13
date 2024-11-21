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
stpwrd <- stop_words %>%
  pull(word) %>%
  str_remove_all('[[:punct:]]')

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

partitions <- train_dtm %>% initial_split(prop = 0.8)

train_data <- training(partitions) %>% 
  select(-c(.id,bclass,mclass))
train_labels <- training(partitions) %>% 
  pull(bclass)

test_data <- testing(partitions) %>% 
  select(-c(.id,bclass,mclass))
test_labels <- testing(partitions) %>% 
  select(.id,bclass,mclass)

x_train <- training(partitions) %>%
  ungroup() %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()

y_train <- training(partitions) %>% 
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
  fit(x = train_data,
      y = train_labels,
      epochs = 20)

#Using logistical NN we have binary_accuracy of 0.5475

#RNN
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

history <- rnn_model %>% fit(
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



#-------------------------------------------------------------------new
# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

# Load required libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(tidymodels)

# Load preprocessed data
load('data/claims-clean-example.RData')

# Partition data into training and testing sets
set.seed(11055)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels_binary <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
train_labels_multi <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels_binary <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
test_labels_multi <- testing(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

# Define vocabulary size and sequence length
vocab_size <- 10000
sequence_length <- 100

# Create and adapt the text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_sequence_length = sequence_length
)
preprocess_layer %>% adapt(train_text)

# Preprocess data
train_sequences <- preprocess_layer(train_text)
test_sequences <- preprocess_layer(test_text)

# Define model for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile binary model
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train binary model
binary_model %>% fit(
  x = train_sequences,
  y = train_labels_binary,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save binary model
save_model_tf(binary_model, "results/binary-model")

# Define model for multi-class classification
multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = length(unique(train_labels_multi)), activation = 'softmax')

# Compile multi-class model
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'sparse_categorical_accuracy'
)

# Train multi-class model
multi_model %>% fit(
  x = train_sequences,
  y = train_labels_multi,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save multi-class model
save_model_tf(multi_model, "results/multi-model")

# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

# Load required libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(tidymodels)

# Load preprocessed data
load('data/claims-clean-example.RData')

# Partition data into training and testing sets
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels_binary <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
train_labels_multi <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels_binary <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
test_labels_multi <- testing(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

# Define vocabulary size and sequence length
vocab_size <- 10000
sequence_length <- 100

# Create and adapt the text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_sequence_length = sequence_length
)
preprocess_layer %>% adapt(train_text)

# Preprocess data
train_sequences <- preprocess_layer(train_text)
test_sequences <- preprocess_layer(test_text)

# Define model for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile binary model
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train binary model
binary_model %>% fit(
  x = train_sequences,
  y = train_labels_binary,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save binary model
save_model_tf(binary_model, "results/binary-model")

# Define model for multi-class classification
multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = length(unique(train_labels_multi)), activation = 'softmax')

# Compile multi-class model
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'sparse_categorical_accuracy'
)

# Train multi-class model
multi_model %>% fit(
  x = train_sequences,
  y = train_labels_multi,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save multi-class model
save_model_tf(multi_model, "results/multi-model")

# Generate predictions
binary_preds <- binary_model %>% predict(test_sequences) %>% round()

multi_preds <- multi_model %>% predict(test_sequences) %>% k_argmax()

# Format predictions into a data frame
class_labels <- levels(factor(claims_clean$mclass))  # Ensure correct class labels

pred_df <- testing(partitions) %>%
  select(.id) %>%
  mutate(
    bclass.pred = ifelse(binary_preds == 1, "Positive", "Negative"),
    mclass.pred = class_labels[as.numeric(multi_preds) + 1]  # Correct indexing
  )

# Export predictions
write.csv(pred_df, "results/example-preds.csv", row.names = FALSE)

# Display results
print(head(pred_df))
