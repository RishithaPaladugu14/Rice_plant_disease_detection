import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report

# Load treatment data
treatment_df = pd.read_csv('rice_disease_treatments.csv')

# Data directories
data_dir = 'rice_leaf_diseases'
img_height, img_width = 150, 150
batch_size = 32

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                        batch_size=batch_size, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                      batch_size=batch_size, class_mode='categorical', subset='validation')

class_names = list(train_gen.class_indices.keys())

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Evaluation
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Prediction with treatment
def predict_and_treat(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_disease = class_names[class_index]

    treatment_row = treatment_df[treatment_df['disease'] == predicted_disease]
    treatment = treatment_row['treatment'].values[0] if not treatment_row.empty else "No treatment found."

    print(f"Predicted Disease: {predicted_disease}")
    print(f"Recommended Treatment: {treatment}")
