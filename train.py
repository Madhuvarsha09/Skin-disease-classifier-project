# src/train.py
from model import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../data/images/train"
val_dir = "../data/images/val"

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 5  # For demo purposes

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

model = create_model(input_shape=(224,224,3), num_classes=train_data.num_classes)
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

model.save("../models/best_model.h5")
print("Model trained and saved at ../models/best_model.h5")
