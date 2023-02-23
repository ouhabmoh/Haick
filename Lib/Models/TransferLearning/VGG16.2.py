import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Set the path to the dataset
train_data_dir = 'path/to/train/dataset'
validation_data_dir = 'path/to/validation/dataset'

# Define the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Use the VGG16 pre-trained model with ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

# Set up callbacks for model checkpoints and early stopping
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Fit the model with the data generators
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size, epochs=10, validation_data=validation_generator, validation_steps=validation_generator.n // batch_size, callbacks=[model_checkpoint, early_stopping])

# Evaluate the model on the validation set
scores = model.evaluate_generator(validation_generator, validation_generator.n // batch_size)

# Print the validation accuracy and loss
print('Validation Accuracy:', scores[1])
print('Validation Loss:', scores[0])

# Save the history, model architecture, and weights
np.save('history.npy', history.history)
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')
