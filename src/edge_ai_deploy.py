# edge_ai_deploy.py
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import precision_score, recall_score

def test_tflite_model(model_path, data_path):
    """Test TensorFlow Lite model on sample data."""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load test data
    df = pd.read_csv(data_path)
    test_images = df["image_path"].values[:10]  # Test on subset
    test_labels = df["label"].values[:10]
    class_names = sorted(df["label"].unique())
    
    predictions = []
    for img_path in test_images:
        img = Image.open(f"data/images/{img_path}").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred = class_names[np.argmax(output)]
        predictions.append(pred)
    
    # Evaluate
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    test_tflite_model("src/model.tflite", "data/recyclable_items_dataset.csv")
