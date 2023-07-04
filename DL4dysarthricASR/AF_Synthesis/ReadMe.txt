			Improving generalization of vocal tract feature reconstruction: 
	from augmented acoustic inversion to articulatory feature reconstruction without articulatory data
				
				Rosanna Turrisi, Raffaele Tavarone, Leonardo Badino

Link to the article: https://ieeexplore.ieee.org/document/8639537

@INPROCEEDINGS{8639537,
  author={Turrisi, Rosanna and Tavarone, Raffaele and Badino, Leonardo},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={Improving Generalization of Vocal Tract Feature Reconstruction: From Augmented Acoustic Inversion to Articulatory Feature Reconstruction without Articulatory Data}, 
  year={2018},
  volume={},
  number={},
  pages={159-166},
  doi={10.1109/SLT.2018.8639537}}

########################################### READ ME #############################################

The codes have been tested on Python 2.7 In order to run, the following Python modules       
are required:

- Numpy/Scipy
- Math
- Sys
- Os
- datetime
- Scikit-learn
- Matplotlib
- Tensorflow 1.0


###############
# Codes      ##
###############

Autoencoder 1
- AE1_config.py: CONFIGURATION FILE containing the functions related to data loading. 
		To run the code, please set up your own configuration file by changing the data paths and/or the file names.
- AE1_model.py: defines the AE1 model described in Sec. 4.1.1
- AE1_train_and_val.py: trains the AE1 model performing early stopping on validation
- AE1_test.py: evaluates the AE1 model on the testing set

Autoencoder 2
- AE2_config.py: CONFIGURATION FILE containing the functions related to data loading. 
		To run the code, please set up your own configuration file by changing the data paths and/or the file names.
- AE2_model.py: defines the AE2 model described in Sec. 4.1.2
- AE2_train_and_val.py: trains the AE2 model performing early stopping on validation
- AE2_test.py: evaluates the AE2 model on the testing set

Residual Network
- ResDNN_config.py: CONFIGURATION FILE containing the functions related to data loading. 
		To run the code, please set up your own configuration file by changing the data paths and/or the file names.
- ResDNN_model.py: defines the ResDNN model described in Sec. 4.2
- ResDNN_train_and_val.py: trains the ResDNN model performing early stopping on validation
- ResDNN_test.py: evaluates the ResDNN model on the testing set


