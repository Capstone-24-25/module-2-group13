source('https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/scripts/package-installs.R')
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))
library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(sparsesvd)
library(Matrix)
library(glmnet)
library(tidymodels)

#tokenization into bigrams
stpwrd <- stop_words %>%
  pull(word) %>%
  str_remove_all('[[:punct:]]')

claims_tokens_long <- claims_clean %>%
  unnest_tokens(output = token, # specifies new column name
                input = text_clean, # specifies column containing text
                token = 'ngrams', n = 2, # how to tokenize
                stopwords = stpwrd) %>% # optional stopword removal
  mutate(token = lemmatize_words(token)) 

#looking at most frequently used bigrams
#claims_tokens_long %>% count(token, sort = TRUE)

#frequency measures based on mclass
claims_tfidf <- claims_tokens_long %>%
  count(mclass, token) %>%
  bind_tf_idf(term = token,
              document = mclass,
              n = n)
claims_tfidf_id <- claims_tfidf %>% 
  left_join(claims_tokens_long %>% select(mclass, .id, bclass, token),
            by = c('mclass','token')) %>% 
  select(mclass, .id, bclass, everything()) %>% 
  unique()

claims_df <- claims_tfidf_id %>%
  pivot_wider(id_cols = c(.id, mclass, bclass),
              names_from = token,
              values_from = tf_idf,
              values_fill = 0) %>% 
  select(mclass, .id, bclass, everything())

#partitioning the data

set.seed(10201024)
partitions <- claims_df %>% initial_split(prop = 0.8)

test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)

#finding the principal components of the training dtm
prcomp_out <- projection_fn(train_dtm, .prop = 0.7)

#fitting the training data to a logistic regression model
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass~., data = train, family= 'binomial')

test_dtm_projection <- reproject_fn(.dtm = test_dtm, prcomp_out)
x_test <- as.matrix(test_dtm_projection)

preds <- predict(fit, 
                 newx = x_test,
                 type = 'response')
