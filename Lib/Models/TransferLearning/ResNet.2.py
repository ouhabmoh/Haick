# Import necessary libraries
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Set path for train, validation and test directories
train_path = 'train/'
valid_path = 'valid/'
test_path = 'test/'

# Set image size
img_width, img_height = 224, 224

# Create a ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers to the base model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Create a new model with the custom top layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for train, validation and test data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(img_width, img_height),
                                                    batch_size=32, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_path, target_size=(img_width, img_height),
                                                    batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(img_width, img_height),
                                                  batch_size=1, class_mode='categorical', shuffle=False)

# Set number of epochs and steps per epoch
epochs = 10
steps_per_epoch = train_generator.n // train_generator.batch_size

# Create a checkpoint to save the best model during training
checkpoint = ModelCheckpoint('resnet_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Train the model
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                    validation_data=valid_generator, validation_steps=valid_generator.n // valid_generator.batch_size,
                    callbacks=[checkpoint])

# Evaluate the model on test data and save results to a CSV file
score = model.evaluate(test_generator, verbose=0)
results = {'loss': score[0], 'accuracy': score[1]}
import pandas as pd
pd.DataFrame(results, index=[0]).to_csv('resnet_results.csv', index=False)

# Save the final trained model
model.save('resnet_final_model.h5')
