test_part <- test_tfidf %>%
  select(-.id)
test_preds_svm_bclass <- test_tfidf %>%
  select(.id)

num_new_columns <- 13770
new_col_names <- paste0("Zero_", 1:num_new_columns)

zero_matrix <- matrix(0, nrow = nrow(test_part), ncol = num_new_columns)
colnames(zero_matrix) <- new_col_names

new_test_part_binary <- cbind(test_part, as.data.frame(zero_matrix))


test_proj_svm_final_binary <- reproject_fn(.dtm = new_test_part_binary, 
                                    proj_out_svmb)

svm_binary_predictions_final <- predict(svm_model, newdata = as.matrix(test_proj_svm_final_binary))


save(svm_binary_predictions_final, file = "../results/test_preds_svm_class_binary.RData")

test_preds_svm_mclass$bclass.pred <- svm_binary_predictions_final

pred_df <- test_preds_svm_mclass

colnames(pred_df)[which(colnames(pred_df) == "pred")] <- "bclass.pred"

save(pred_df, file = "../results/preds-group13.RData")
