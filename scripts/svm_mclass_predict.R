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

# save(test_tfidf, file = "../data/claims_clean_test.RData")

load('../data/claims_clean_test.RData')

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

# PC
test_part <- test_tfidf %>%
  select(-.id)
test_preds_svm_mclass <- test_tfidf %>%
  select(.id)

test_proj_svm_final <- reproject_fn(.dtm = new_test_part, 
                                    proj_out_svm_m)

num_new_columns <- 13770
new_col_names <- paste0("Zero_", 1:num_new_columns)

zero_matrix <- matrix(0, nrow = nrow(test_part), ncol = num_new_columns)
colnames(zero_matrix) <- new_col_names

new_test_part <- cbind(test_part, as.data.frame(zero_matrix))

svm_predictions_final <- predict(svm_model_m, newdata = as.matrix(test_proj_svm_final))

test_preds_svm_mclass$pred <- svm_predictions_final

save(test_preds_svm_mclass, file = "../results/svm_mclass_preds.RData")
