library(keras)
library(tensorflow)

train_dir <- "C:/Users/erikl_xzy542i/Documents/Master_local/T3/MachineLearning/pneumonia/chest_xray/train"
val_dir <- "C:/Users/erikl_xzy542i/Documents/Master_local/T3/MachineLearning/pneumonia/chest_xray/val"
test_dir <- "C:/Users/erikl_xzy542i/Documents/Master_local/T3/MachineLearning/pneumonia/chest_xray/test"

# Data generators
train_datagen <- image_data_generator(
  rescale = 1/255,          # Normalize pixel values (0-1)
  rotation_range = 20,      # Augment: Randomly rotate images
  width_shift_range = 0.2,  # Augment: Horizontal shifts
  height_shift_range = 0.2, # Augment: Vertical shifts
  zoom_range = 0.2,         # Augment: Random zoom
  horizontal_flip = TRUE    # Augment: Horizontal flips
)

val_test_datagen <- image_data_generator(
  rescale = 1/255 # Normalize pixel values for validation/testing
)

# Training generator
train_generator <- flow_images_from_directory(
  train_dir,
  generator = train_datagen,
  target_size = c(224, 224),  # Resize images to match model input size
  batch_size = 32,           # Number of images per batch
  class_mode = "binary"      # Binary classification (normal vs pneumonia)
)

# Validation generator
val_generator <- flow_images_from_directory(
  val_dir,
  generator = val_test_datagen,
  target_size = c(224, 224),
  batch_size = 16,
  class_mode = "binary"
)

# Test generator
test_generator <- flow_images_from_directory(
  test_dir,
  generator = val_test_datagen,
  target_size = c(224, 224),
  batch_size = 32,
  class_mode = "binary",
  shuffle = FALSE            # Do not shuffle for testing (to align predictions with labels)
)


# Check class indices
print(train_generator$class_indices)

# Example: Visualize a batch of images
batch <- generator_next(train_generator)
par(mfrow = c(2, 3))  # Set plot grid (4 rows, 8 columns)
for (i in 1:6) {
  img <- batch[[1]][i,,,]  # Extract image
  plot(as.raster(img))
}
par(mfrow=c(1,1))


################################################################################
###############################   Train models  ################################
################################################################################

# Basic CNN as our first model
model <- keras_model_sequential() %>%
  # Convolutional layer 1
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(224, 224, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Convolutional layer 2
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Convolutional layer 3
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.2)%>%
  
  # Flatten and fully connected layers
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") # Output layer for binary classification


model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


history <- model %>% fit(
  train_generator,
  steps_per_epoch = as.integer(train_generator$samples / train_generator$batch_size),
  epochs = 10,
  validation_data = val_generator,
  validation_steps = as.integer(val_generator$samples / val_generator$batch_size)
)


results <- model %>% evaluate(
  test_generator,
  steps = as.integer(test_generator$samples / test_generator$batch_size)
)

print(results)

