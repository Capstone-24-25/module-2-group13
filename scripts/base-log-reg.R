library(tidyverse)
library(tidytext)
library(tidymodels)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

#loading example clean data
load('../data/claims-clean-example.RData')

#loading projection functions
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

#tokenization
claims_tfidf <- claims_clean %>% 
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

#data split
set.seed(3)

split <- initial_split(claims_tfidf, 
                       prop = 0.7,
                       strata = 'bclass')

#separating labels
train_dtm <- training(split) %>% 
  select(-.id, -bclass, -mclass)
train_labels <- training(split) %>% 
  select(.id, bclass, mclass)

test_dtm <- testing(split) %>% 
  select(-.id, -bclass, -mclass)
test_labels <- testing(split) %>% 
  select(.id, bclass, mclass)

#projection
proj_out <- projection_fn(.dtm = train_dtm, 
                          .prop = 0.7)
train_dtm_proj <- proj_out$data

save(proj_out, file = "../results/base-logreg-model/proj-out.RData")

#regression
train <- train_labels %>% 
  transmute(bclass = factor(bclass)) %>% 
  bind_cols(train_dtm_proj)

fit <- glm(bclass ~ ., 
           data = train,
           family = binomial)

#save model
save(fit, file = "../results/base-logreg-model/base-logreg-model.RData")

test_proj <- reproject_fn(.dtm = test_dtm, 
                          proj_out)

#get predictions
preds <- predict(fit,
                 newdata = as.data.frame(test_proj),
                 type = 'response')

pred_df <- test_labels %>% 
  transmute(bclass = factor(bclass)) %>% 
  bind_cols(pred = as.numeric(preds)) %>% 
  mutate(bclass.pred = factor(pred > 0.5,
                              labels = levels(bclass)))

#save pred_df
base_pred_df <- pred_df
save(base_pred_df, file = '../results/base-logreg-model/base-logreg-preds.RData')

#metrics
class_metrics = metric_set(sensitivity,
                           specificity,
                           accuracy,
                           roc_auc)

metrics <- pred_df %>% 
  class_metrics(truth = bclass,
                estimate = bclass.pred,
                pred,
                event_level = 'second')

#save metrics
base_metrics <- metrics
save(base_metrics, file = '../results/base-logreg-model/base_metrics.RData')
