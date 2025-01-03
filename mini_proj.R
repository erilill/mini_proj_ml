# Authors: Erik Lillrank, Mark Becker & Felix Andersson

# Load libraries
library(keras)
library(tensorflow)
library(yardstick)

# Set seed
set.seed(1337)  # R seed
tensorflow::set_random_seed(1337)  # TensorFlow seed
Sys.setenv("PYTHONHASHSEED" = 1337)  # Python hash seed

# Set the directories for the data
# Data is available at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Note that we have moved the images from the validation set to the training set
train_dir <- "C:/Users/erikl_xzy542i/Documents/Master_local/T3/MachineLearning/pneumonia/chest_xray/train"
test_dir <- "C:/Users/erikl_xzy542i/Documents/Master_local/T3/MachineLearning/pneumonia/chest_xray/test"

# Data generators
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = 'nearest',
  validation_split = 0.2
)

val_test_datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

# Training generator
train_generator <- flow_images_from_directory(
  train_dir,
  generator = train_datagen,
  target_size = c(128, 128), 
  batch_size = 32,           
  class_mode = "binary",
  subset = "training"
)

# Validation generator
val_generator <- flow_images_from_directory(
  train_dir,
  generator = val_test_datagen,
  target_size = c(128, 128),
  batch_size = 16,
  class_mode = "binary",
  shuffle = FALSE,
  subset = "validation"
)

# Check for overlap between train and validation sets
intersect(train_generator$filepaths, val_generator$filepaths) # If 0, no overlap

# Test generator
test_generator <- flow_images_from_directory(
  test_dir,
  generator = val_test_datagen,
  target_size = c(128, 128),
  batch_size = 8,
  class_mode = "binary",
  shuffle = FALSE
)


# Check class indices
print(train_generator$class_indices)

# Example: Visualize a batch of images
batch <- generator_next(train_generator)
par(mfrow = c(2, 3))  
for (i in 1:6) {
  img <- batch[[1]][i,,,]  
  plot(as.raster(img))
}
par(mfrow=c(1,1))


################################################################################
###############################   Train models  ################################
################################################################################

# Basic CNN as our first model
model <- keras_model_sequential() %>%
  # Convolutional layer 1
  layer_conv_2d(filters = 46, kernel_size = c(3, 3), activation = "relu", input_shape = c(128, 128, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Convolutional layer 2
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Convolutional layer 3
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Flatten and fully connected layers
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu", kernel_regularizer = regularizer_l2(0.1)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") 

# Early stopping scheme
callback_early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE
)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 5e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


history <- model %>% fit(
  train_generator,
  steps_per_epoch = as.integer(train_generator$samples / train_generator$batch_size),
  epochs = 50,
  validation_data = val_generator,
  validation_steps = as.integer(val_generator$samples / val_generator$batch_size),
  callbacks = list(callback_early_stop)
)

plot(history)
tail(history$metrics$val_loss,1)
tail(history$metrics$val_accuracy,1)

################################################################################
#############################   Transfer models   ##############################
################################################################################

################################  densenet121  #################################
# We use the pre trained model densenet 121
base_model <- application_densenet121(
  weights = "imagenet",  
  include_top = FALSE,  
  input_shape = c(128, 128, 3)
)

# Freeze layers of the pre trained model
freeze_weights(base_model)

# Add top layers
model2 <- keras_model_sequential() %>%
  base_model %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu', kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')  


model2 %>% compile(optimizer_adam(learning_rate = 1e-5), loss = 'binary_crossentropy', metrics = c('accuracy'))

# Early stopping scheme
callback_early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE
)

# Unfreeze trainable layers
model2$trainable <- TRUE

# Fine-tune the model to data
history_fine <- model2 %>% fit(
  train_generator,
  steps_per_epoch = train_generator$samples %/% train_generator$batch_size,
  epochs = 50,
  validation_data = val_generator,
  validation_steps = val_generator$samples %/% val_generator$batch_size,
  callbacks = list(callback_early_stop)
)
plot(history_fine)
tail(history_fine$metrics$val_loss,1)
tail(history_fine$metrics$val_accuracy,1)


##################################  VGG16  #####################################
# We use the pre trained model VGG16
base_model2 <- application_vgg16(
  weights = "imagenet",  
  include_top = FALSE,  
  input_shape = c(128, 128, 3)
)

# Freeze layers of the pre trained model
freeze_weights(base_model2)

# Add top layers
model3 <- keras_model_sequential() %>%
  base_model2 %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu', kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')  


model3 %>% compile(optimizer_adam(learning_rate = 1e-5), loss = 'binary_crossentropy', metrics = c('accuracy'))

# Early stopping scheme
callback_early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE
)

# Unfreeze trainable layers
model3$trainable <- TRUE

# Fine-tune the model to data
history_fine2 <- model3 %>% fit(
  train_generator,
  steps_per_epoch = train_generator$samples %/% train_generator$batch_size,
  epochs = 50,
  validation_data = val_generator,
  validation_steps = val_generator$samples %/% val_generator$batch_size,
  callbacks = list(callback_early_stop)
)
plot(history_fine2)
tail(history_fine2$metrics$val_loss,1)
tail(history_fine2$metrics$val_accuracy,1)


################################################################################
############################   Model Evaluation   ##############################
################################################################################

# Here we compute accuracy, confusion matrices, AUC, etc. on the test data

# True labels in the test set
true_labels <- factor(test_generator$classes)
true_labels_df <- data.frame(truth = factor(true_labels))

################################  Model 1  #####################################

# Make predictions on the test data
predictions <- model %>% predict(test_generator, steps = as.integer(test_generator$samples / test_generator$batch_size))
predicted_classes <- factor(ifelse(predictions > 0.5, 1, 0))

results_df <- data.frame(truth = true_labels, pred = predicted_classes)
results_df_prob <- data.frame(truth = true_labels, pred = predictions)

# 1. Compute Accuracy
acc <- accuracy(results_df, truth, pred, event_level = "second")
print(acc)

# 2. Compute AUC
auc_result <- roc_auc(results_df_prob, truth, pred, event_level = "second")
print(auc_result)

# 3. Compute Precision
precision_result <- precision(results_df, truth, pred, event_level = "second")
print(precision_result)

# 4. Compute Recall (Sensitivity)
recall_result <- recall(results_df, truth, pred, event_level = "second")
print(recall_result)

# 5. Compute F1-Score
f1_result <- f_meas(results_df, truth, pred, event_level = "second")
print(f1_result)

################################  Model 2  #####################################

# Make predictions on the test data
predictions2 <- model2 %>% predict(test_generator, steps = as.integer(test_generator$samples / test_generator$batch_size))
predicted_classes2 <- factor(ifelse(predictions2 > 0.5, 1, 0))

results_df2 <- data.frame(truth = true_labels, pred = predicted_classes2)
results_df_prob2 <- data.frame(truth = true_labels, pred = predictions2)

# 1. Compute Accuracy
acc2 <- accuracy(results_df2, truth, pred, event_level = "second")
print(acc2)

# 2. Compute AUC
auc_result2 <- roc_auc(results_df_prob2, truth, pred, event_level = "second")
print(auc_result2)

# 3. Compute Precision
precision_result2 <- precision(results_df2, truth, pred, event_level = "second")
print(precision_result2)

# 4. Compute Recall (Sensitivity)
recall_result2 <- recall(results_df2, truth, pred, event_level = "second")
print(recall_result2)

# 5. Compute F1-Score
f1_result2 <- f_meas(results_df2, truth, pred, event_level = "second")
print(f1_result2)

################################  Model 3  #####################################

# Make predictions on the test data
predictions3 <- model3 %>% predict(test_generator, steps = as.integer(test_generator$samples / test_generator$batch_size))
predicted_classes3 <- factor(ifelse(predictions3 > 0.5, 1, 0))

results_df3 <- data.frame(truth = true_labels, pred = predicted_classes3)
results_df_prob3 <- data.frame(truth = true_labels, pred = predictions3)

# 1. Compute Accuracy
acc3 <- accuracy(results_df3, truth, pred, event_level = "second")
print(acc3)

# 2. Compute AUC
auc_result3 <- roc_auc(results_df_prob3, truth, pred, event_level = "second")
print(auc_result3)

# 3. Compute Precision
precision_result3 <- precision(results_df3, truth, pred, event_level = "second")
print(precision_result3)

# 4. Compute Recall (Sensitivity)
recall_result3 <- recall(results_df3, truth, pred, event_level = "second")
print(recall_result3)

# 5. Compute F1-Score
f1_result3 <- f_meas(results_df3, truth, pred, event_level = "second")
print(f1_result3)

################################################################################
########################   Save models for plotting   ##########################
################################################################################

# We save the models in HDF5 format and then use Netron to do the visualization
model %>% save_model_hdf5("model.h5")
model2 %>% save_model_hdf5("model2.h5")
model3 %>% save_model_hdf5("model3.h5")
