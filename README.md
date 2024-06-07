# Tensorflower

## Training
- Server start: ``` python3 server.py ```
- Crossval server start: ``` python3 server_crossval.py ```
- Full-dataset augmented client: ``` python3 client_augmented_fulldataset.py ```
- Full-dataset client: ``` python3 client_fulldataset.py ```
- Subdivided augmented client: ``` python3 client_augmented_csv.py path_to_train.csv path_to_validation.csv ```
- Full-dataset augmented crossval client: ``` python3 client_augmented_fulldataset_crossval.py fold_number ``` (fold_number starts at 1)
- Subdivided augmented crossval client: ``` python3 client_augmented_csv_crossval.py data.csv fold_number ``` (fold_number starts at 1)
- Vision Transformer full-dataset augmented client: ``` python3 client_augmented_csv_visiontransformer.py path_to_train.csv path_to_validation.csv ```
- HuggingFace Client:  ``` python3 client_huggingface.py client_str train.csv val.csv ``` (client_str is a client identifier)

## Testing
- Centralized Test: ``` python3 test_specific_model_csv.py path_to_model.npz path_to_test.csv ```
- Centralized Test (hugging face): ``` python3 test_huggingface_model.py model_path.npz test_path.csv ```

## Credits
Dataset from: https://www.tensorflow.org/datasets/catalog/colorectal_histology

Download link: https://zenodo.org/records/53169#.XGZemKwzbmG
