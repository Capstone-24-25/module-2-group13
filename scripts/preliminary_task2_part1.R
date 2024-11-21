library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(caret)
library(glmnet)
library(pROC)

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
#####
set.seed(102722)
partitions <- text_word %>% initial_split(text_word, 
                                          prop = 0.7,
                                          strata = 'bclass')

test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)

proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

test_proj <- reproject_fn(.dtm = test_dtm, 
                          proj_out)

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

logistic_model <- glm(bclass ~ ., data = train, family = "binomial")

predictions <- predict(logistic_model, newdata = as.data.frame(test_proj), type = "response")

roc_curve <- roc(test_labels$bclass, predictions)
auc(roc_curve) #Area under the curve: 0.8344

