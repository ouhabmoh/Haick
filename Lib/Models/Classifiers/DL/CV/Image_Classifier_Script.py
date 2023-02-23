# classifier.py
import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hydra

@hydra.main(config_path="configs", config_name="classifier.yaml")
def main(cfg):
    image_dir = hydra.utils.to_absolute_path(cfg.image_dir)
    epochs = cfg.epochs
    batch_size = cfg.batch_size

    # Split data into training and testing sets
    train_dir = os.path.join(image_dir, "train")
    test_dir = os.path.join(image_dir, "test")
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')

    # Define models
    models = [
        ("ResNet", ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
        ("VGG", VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
        ("Inception", InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    ]

    # Train models and save to file
    model_dir = "Image_Classifiers_Models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for name, model in models:
        base_model_output = model.output
        x = GlobalAveragePooling2D()(base_model_output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(train_generator.num_classes, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs,
                  validation_data=test_generator)
        model_path = os.path.join(model_dir, name + ".h5")
        model.save(model_path)

    # Evaluate models and log results to file
    eval_file = "results/evaluations.txt"
    if not os.path.exists("results"):
        os.makedirs("results")

    with open(eval_file, 'w') as f:
        for name, model in models:
            model_path = os.path.join(model_dir, name + ".h5")
            model = load_model(model_path)
            y_pred = model.predict(test_generator).argmax(axis=-1)
            y_test = test_generator.classes
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            f.write("{}:\nAccuracy: {}\nConfusion Matrix:\n{}\n".format(name, acc, cm))

