import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path


# Charger le modèle
model = tf.keras.models.load_model("final_modell.keras")

# Définir la taille d'image
img_height = 200
img_width = 200

# Charger et préparer l'image
img_path = "/home/mactar/Downloads/img3.jpg"
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Ajoute une dimension batch

# Prédiction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Récupérer les noms des classes
data_dir = "reconn_image"
data_dir = Path(data_dir)
train_path = data_dir / "train"

temp_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=(200, 200),
    batch_size=1  # Peu importe ici
)

class_names = temp_data.class_names
print(f"Classe prédite : {class_names[predicted_class]}")
print(f"Probabilités : {predictions[0]}")
