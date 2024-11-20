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


# extract headings with text function
text_heading_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}


parse_headings_text <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_headings_clean = text_heading_fn(text_tmp)) %>%
    unnest(text_headings_clean) 
  return(out)
}

#loading claims data from example
load('../data/claims-clean-example.RData')

#creating combined column
claims_clean <- claims_clean %>%
  parse_headings_text()

# save dataset
save(claims_clean, file = "../data/claims-clean-task-1.RData")
load('../data/claims-clean-task-1.RData')

# tokening
heading_tfidf <- claims_clean %>% 
  unnest_tokens(output = token,
                input = text_headings_clean,
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

split <- initial_split(heading_tfidf, 
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

#load projection functions
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

#projection
proj_out <- projection_fn(.dtm = train_dtm, 
                          .prop = 0.7)
train_dtm_proj <- proj_out$data

#regression
train <- train_labels %>% 
  transmute(bclass = factor(bclass)) %>% 
  bind_cols(train_dtm_proj)

fit <- glm(bclass ~ ., 
           data = train,
           family = binomial)

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
headings_pred_df <- pred_df
save(headings_pred_df, file = '../results/headings-logreg-preds.RData')

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
headings_metrics <- metrics
save(headings_metrics, file = '../results/headings_metrics.RData')