from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("cifar10_cnn.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        if file:
            # Save and preprocess image
            image_path = os.path.join("static", file.filename)
            file.save(image_path)
            image = Image.open(image_path).resize((32, 32))
            image = np.array(image).astype("float32") / 255.0
            if image.shape != (32, 32, 3):
                return render_template("index.html", error="Image must be 32x32 RGB.")
            image = np.expand_dims(image, axis=0)

            # Predict
            pred = model.predict(image)
            class_id = np.argmax(pred)
            prediction = class_names[class_id]
            confidence = f"{pred[0][class_id] * 100:.2f}%"

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)

