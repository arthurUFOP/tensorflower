import numpy as np
import tensorflow as tf
import flwr as fl
import os
from sklearn.model_selection import train_test_split
from skimage.io import imread

physical_devices = tf.config.list_physical_devices('GPU')
for phy_dev in physical_devices:
  tf.config.experimental.set_memory_growth(phy_dev, True)

TESTING    = False
TRAIN_SIZE = 0.8
HOST       = "192.168.0.39"
PORT       = "7517"

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, model, base_model_ref, lr=0.001, local_epochs=10, batch_size=128, verbose=1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.base_model_ref = base_model_ref
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Learning rate decay
        if bool(config["lr_decay"]):
          self.lr *= float(config["decay_factor"])
          self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])

        # Freeze
        if bool(config["alter_trainable"]):
          self.base_model_ref.trainable = bool(config["trainable"])

        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=self.local_epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, auc, prec, rec = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": float(accuracy), "auc":float(auc), "precision":float(prec), "recall":float(rec)}

# Returns the model + a reference to base model (usefull for freezing)
def gen_resnet50_model():
  inputs = tf.keras.Input((150,150,3))
  model = tf.keras.applications.resnet.ResNet50(input_tensor=inputs, classes=8, weights=None)
  return tf.keras.Model(inputs, model.output), model

def gen_effnetb0_model():
  inputs = tf.keras.Input((150,150,3))
  model = tf.keras.applications.EfficientNetB0(input_tensor=inputs, classes=8, weights=None)
  return tf.keras.Model(inputs, model.output), model

def load_images_from_directory(directory):
    images = []
    labels = []
    counter = -1

    for subdir, _, files in os.walk(directory):
        lbl = np.zeros((8,), dtype=np.int64)
        lbl[counter] = 1
        counter+=1
        for file in files:
            file_path = os.path.join(subdir, file)
            image = imread(file_path)
            
            images.append(image)
            labels.append(lbl)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

print("Reading Data...")
features, labels = load_images_from_directory("Kather_texture_2016_image_tiles_5000")

if TESTING:
  assert features.shape == (5000, 150, 150, 3)
  assert labels.shape == (5000, 8)

x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=TRAIN_SIZE, random_state=7585)
print("Finished Reading Data!")

#model, base_model = gen_resnet50_model()
model, base_model = gen_effnetb0_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])
# Starting Client

print("Starting client...\n\n")
fl.client.start_numpy_client(server_address=f"{HOST}:{PORT}", client=FlowerClient(x_train, y_train, x_test, y_test, model, base_model))
print("\n\nTraining Done!")
