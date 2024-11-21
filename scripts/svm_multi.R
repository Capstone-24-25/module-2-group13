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

# use clean dataset with text
load("./data/claims-clean-example.RData")

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

# tokenize
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

# PC
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
accuracy_svm_m <- sum(diag(confusion_matrix_svm_m)) / sum(confusion_matrix_svm_m) 

#0.767285

classes_svm_m <- levels(test_labels_svm_m$mclass)

metrics_svm_m <- sapply(classes_svm_m, function(cls) {
  # Extract confusion matrix values
  TP <- confusion_matrix_svm_m[cls, cls]
  FN <- sum(confusion_matrix_svm_m[, cls]) - TP
  FP <- sum(confusion_matrix_svm_m[cls, ]) - TP
  TN <- sum(confusion_matrix_svm_m) - TP - FN - FP
  
  # Sensitivity and Specificity
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  return(c(Sensitivity = sensitivity, Specificity = specificity))
})

# Convert to a data frame for better readability
metrics_df <- as.data.frame(t(metrics_svm_m))

#save model
save(svm_model_m, file = "./results/svm_multi/svm_multi_model.RData")
save(metrics_df, file = "../results/svm_multi/sensitivity_specificity.RData")
save(accuracy_svm_m, file = "../results/svm_multi/accuracy_svm_m.RData")
