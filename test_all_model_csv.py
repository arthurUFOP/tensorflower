import numpy as np
import tensorflow as tf
import flwr as fl
import os
import sys
import csv
from sklearn.model_selection import train_test_split
from skimage.io import imread

if len(sys.argv)!=3:
    print(f"ERROR: Wrong usage!\nCorrect one: python3 {sys.argv[0]} 'path_to_model_dir' 'test_file.csv' ")
    exit(1)

def gen_effnetb0_model():
  inputs = tf.keras.Input((150,150,3))
  model = tf.keras.applications.EfficientNetB0(input_tensor=inputs, classes=8, weights=None)
  return tf.keras.Model(inputs, model.output), model

def load_images_from_csv(csv_path):
    images = []
    labels = []

    header = True
    with open(csv_path) as f:
      reader = csv.reader(f)
      for row in reader:

        if header:
          header = False
          continue
        
        images.append(imread(row[0])) # Feature

        label = np.zeros((8,), dtype=np.int64) # Label
        label[int(row[1])] = 1
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

path = sys.argv[1]

print("Reading Data...")
x_test, y_test = load_images_from_csv(sys.argv[2])
print("Finished Reading Data!")

model, base_model = gen_effnetb0_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy",
                                      tf.keras.metrics.AUC(),
                                      tf.keras.metrics.Precision(),
                                      tf.keras.metrics.Recall()])

best_model = ''
best_results = (-1, -1, -1, -1, -1)

for i, file in enumerate(os.listdir(path)):
    if os.path.splitext(file)[1] != ".npz":
        continue

    data = np.load(os.path.join(path, file))
    weights = []

    for key in data.keys():
        weights.append(data[key])
    model.set_weights(weights)
    loss, accuracy, auc, prec, rec = model.evaluate(x_test, y_test, verbose=1)
    
    if accuracy > best_results[1]:
        best_results = (loss, accuracy, auc, prec, rec)
        best_model = file

print(f"BEST MODEL: {best_model}")
print(f"LOSS = {best_results[0]}\nACC = {best_results[1]}\nAUC = {best_results[2]}\nPREC = {best_results[3]}\nREC = {best_results[4]}\n")
