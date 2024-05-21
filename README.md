# Tensorflower

## Training
- Server start: ``` python3 server.py ```
- Full-dataset augmented client: ``` python3 client_augmented_fulldataset.py ```
- Full-dataset client: ``` python3 client_fulldataset.py ```
- Subdivided client: ``` python3 client_augmented_csv.py path_to_train.csv path_to_validation.csv ```

## Testing
- Centralized Test: ``` python3 test_specific_model_csv.py path_to_model.npz path_to_test.csv ```

## Credits
Dataset from: https://www.tensorflow.org/datasets/catalog/colorectal_histology

Download link: https://zenodo.org/records/53169#.XGZemKwzbmG
