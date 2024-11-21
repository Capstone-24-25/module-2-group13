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


####svm bclass
set.seed(123)
partitions_svmb <- text_word %>% initial_split(text_word, 
                                          prop = 0.7,
                                          strata = 'bclass')

test_dtm_svmb <- testing(partitions_svmb) %>%
  select(-.id, -bclass, -mclass)
test_labels_svmb <- testing(partitions_svmb) %>%
  select(.id, bclass, mclass)

train_dtm_svmb <- training(partitions_svmb) %>%
  select(-.id, -bclass, -mclass)
train_labels_svmb <- training(partitions_svmb) %>%
  select(.id, bclass, mclass)

# PC
proj_out_svmb <- projection_fn(.dtm = train_dtm_svmb, .prop = 0.7)
train_dtm_projected_svmb <- proj_out_svmb$data

test_proj_svmb <- reproject_fn(.dtm = test_dtm_svmb, 
                          proj_out_svmb)

train_svmb <- train_labels_svmb %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected_svmb)

# svm
svm_model <- svm(bclass ~ ., data = train_svmb, kernel = "radial", cost = 1, gamma = 0.1)

svm_predictions <- predict(svm_model, newdata = as.matrix(test_proj_svmb))
confusion_matrix_svmb <- table(Predicted = svm_predictions, Actual = test_labels_svmb$bclass)
accuracy_svmb <- sum(diag(confusion_matrix_svmb)) / sum(confusion_matrix_svmb) 


TP <- confusion_matrix_svmb["Relevant claim content", "Relevant claim content"]  # True Positives
FN <- confusion_matrix_svmb["N/A: No relevant content.", "Relevant claim content"]  # False Negatives
TN <- confusion_matrix_svmb["N/A: No relevant content.", "N/A: No relevant content."]  # True Negatives
FP <- confusion_matrix_svmb["Relevant claim content", "N/A: No relevant content."]  # False Positives

sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)

cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")

#save model
save(svm_model, file = "./results/svm_binary/svm_binary_model.RData")

##Sensitivity: 0.7863777
##Specificity: 0.866171 
##accuracy: 0.8226351

acc_df <- data.frame(
  model = "SVM_bclass",
  accuracy = accuracy_svmb,
  Sensitivity = sensitivity,
  Specificity = specificity
)

save(acc_df, file = "../results/svm_binary/accuracy_svm_b.RData")
