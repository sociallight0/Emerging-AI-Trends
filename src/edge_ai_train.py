# edge_ai_train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def train_edge_ai_model(data_path, output_model_path):
    """Train and convert a MobileNetV2 model to TensorFlow Lite."""
    # Load dataset
    df = pd.read_csv(data_path)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Prepare data
    train_generator = datagen.flow_from_dataframe(
        df, directory="data/images", x_col="image_path", y_col="label",
        target_size=(224, 224), batch_size=32, subset="training"
    )
    val_generator = datagen.flow_from_dataframe(
        df, directory="data/images", x_col="image_path", y_col="label",
        target_size=(224, 224), batch_size=32, subset="validation"
    )
    
    # Load and fine-tune MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation="softmax")  # Adjust for number of classes
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, validation_data=val_generator, epochs=5)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save model
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved to {output_model_path}")

if __name__ == "__main__":
    train_edge_ai_model("data/recyclable_items_dataset.csv", "src/model.tflite")
