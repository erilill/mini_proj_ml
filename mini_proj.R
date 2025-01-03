library(keras)
library(tensorflow)

set.seed(1337)  # R seed
tensorflow::set_random_seed(1337)  # TensorFlow seed
Sys.setenv("PYTHONHASHSEED" = 1337)  # Python hash seed

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
  batch_size = 32,
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

results <- model %>% evaluate(
  test_generator,
  steps = as.integer(test_generator$samples / test_generator$batch_size)
)

print(results)


################################################################################
#############################   Transfer models   ##############################
################################################################################

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

# Evaluate against test data
results2 <- model2 %>% evaluate(
  test_generator,
  steps = as.integer(test_generator$samples / test_generator$batch_size)
)
print(results2)


################################################################################
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

# Evaluate against test data
results3 <- model3 %>% evaluate(
  test_generator,
  steps = as.integer(test_generator$samples / test_generator$batch_size)
)
print(results3)

