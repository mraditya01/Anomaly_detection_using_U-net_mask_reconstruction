# Anomaly_detection_using_U-net_mask_reconstruction
**This is a model for Unsupervised Detection of Anomalous Sounds in Driving Environment**. 
<p align="center">
  <img src="https://user-images.githubusercontent.com/59830001/195421530-6a37ceea-5db0-4752-bad3-2edd56238dfa.png" width="50%" height="50%"/>
</p> 
More indepth explanation about the research can be seen in the Google Drive PDFs[1]: https://drive.google.com/file/d/1zBpyDxoOHeXSF7g31pDpC08_Xj-PLT7r/view?usp=sharing.


## Description
The system consists of two main scripts:
- `00_train.py`
  - This script trains models for each Machine Type by using the directory **dev_data/<Machine_Type>/train/** or **eval_data/<Microphone_Number>/train/**.
- `01_test.py`
  - This script makes csv files for each Machine ID including the anomaly scores for each wav file in the directory **dev_data/<Microphone_Number>/test/** or **eval_data/<Microphone_Number>/test/**.
  - The csv files will be stored in the directory **result/**.
  - If the mode is "development", it also makes the csv files including the AUC and pAUC for each Machine ID. 
- `train.ipynb`
  - This script trains models for one Machine Type of your choice by using the directory **dev_data/<Machine_Type>/train/** or **eval_data/<Machine_Type>/train/**.

## Usage

### 1. Clone repository
Clone this repository from Github.

### 2. Dataset structure
Unzip the downloaded files and make the directory structure as follows:
- ./Anomaly_detection_using_U-net_mask_reconstruction
    - /dev_data
        - /Microphone 1
            - /train (Only normal data for all Machine IDs are included.)
                - /normal_id_00_00000000.wav
                - ...
                - /normal_id_00_00000999.wav
            - /test (Normal and anomaly data for all Machine IDs are included.)
                - /normal_id_00_00000000.wav
                - ...
                - /normal_id_00_00000349.wav
                - /anomaly_id_00_00000000.wav
                - ...
                - /anomaly_id_00_00000263.wav

        - /Microphone 2 (The other Microphone Types must have the same directory structure as Microphone 1.)
        - /Microphone 3
        - /Microphone 4
        - /Microphone 5
        - .
        - .
        - /Microphone 8
    - /00_train.py
    - /01_test.py
    - /common.py
    - /keras_model.py
    - /baseline.yaml
    - /readme.md

### 3. Change parameters
You can change the parameters for feature extraction and model definition by editing `baseline.yaml`.

### 4. Run training script (for development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset **dev_data/<Microphone_Number>/train/**.
```
$ python 00_train.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for "development"                                       |  
| `-e`                        | `--eval`                          | Mode for "evaluation"                                        | 

`00_train.py` trains the models for each Machine Type and saves the trained models in the directory **model/**.

### 5. Run test script (for development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset **dev_data/<Machine_Type>/test/**.
```
$ python 01_test.py -d
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates the anomaly scores for each wav file in the directory **dev_data/<Machine_Type>/test/**.
The csv files for each Microphone_Number including the anomaly scores will be stored in the directory **result/**.
If the mode is "development", the script also makes the csv files including the AUCs and pAUCs for each Machine ID. 

### 6. Check results
You can check the anomaly scores in the csv files in the directory **result/**.
Each anomaly score corresponds to a wav file in the directory **dev_data/<Machine_Type>/test/**:

`anomaly_score_Microphone1_id_00.csv`
```  
normal_id_01_00000000.wav	6.95342025
normal_id_01_00000001.wav	6.363580014
normal_id_01_00000002.wav	7.048401741
normal_id_01_00000003.wav	6.151557502
normal_id_01_00000004.wav	6.450118248
normal_id_01_00000005.wav	6.368985477
  ...
```

Also, you can check the AUC and pAUC scores for each Machine ID:

`result.csv`
```

## Dependency
We develop the source code on Windows10.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.1.6
- Keras-Applications            == 1.0.8
- Keras-Preprocessing           == 1.0.5
- matplotlib                    == 3.0.3
- numpy                         == 1.16.0
- PyYAML                        == 5.1
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
- tqdm                          == 4.23.4
