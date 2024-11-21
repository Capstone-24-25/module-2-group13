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

bigram_tf_idf <- claims_clean %>%
  unnest_tokens(output = token, # specifies new column name
                input = text_clean, # specifies column containing text
                token = 'ngrams', n = 2, # how to tokenize
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

#partitioning the data
set.seed(10201024)
partitions <- bigram_tf_idf %>% initial_split(prop = 0.8)

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
train_dtm_projected <- prcomp_out$data
#fitting the training data to a logistic regression model
prcomp_out$n_pc

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass~., data = train, family = 'binomial')

test_dtm_projected <- reproject_fn(.dtm = test_dtm, prcomp_out)
x_test <- as.matrix(test_dtm_projected)

reg_preds <- predict(fit, newx = x_test, type = 'response')
reg_pred_df <- test_labels %>% 
  transmute(bclass = as.factor(bclass)) %>% 
  bind_cols(pred = as.numeric(reg_preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

#activity2
# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

# choose a constrait strength by cross-validation
set.seed(102722)
cvout <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

# view results
cvout

# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, prcomp_out)

# coerce to matrix
x_test <- as.matrix(test_dtm_projected)

# compute predicted probabilities
preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'response')

pred_df <- test_labels %>% 
  transmute(bclass = as.factor(bclass)) %>% 
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

results <- pred_df %>% 
  panel(truth = bclass, estimate = bclass.pred, pred, event_level = 'second')

save(results, file = "/Users/henrylouie/Downloads/module-2-group13/results/model_result.RData")

getwd()
