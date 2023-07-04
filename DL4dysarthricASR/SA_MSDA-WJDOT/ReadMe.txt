########################################### READ ME #############################################

The codes have been tested on Python 3.7.6. In order to run, the following Python modules
are required:

- Numpy/SciPy
- Matplotlib
- sys, os, glob
- PyTorch and dependencies from http://pytorch.org.
- POT python package


#############
#   Codes  ##
#############

- BLSTM_models.py: it contains the BLSTM model for single- and multi-task learning
- BLSTM_utilities.py: it contains useful function related to data_loading, training, inference and visualization
- AllSpeak_SI_model.py: it trains the Speaker-Independent (SI) model. It takes three command-line arguments:
    saver_dir: Path to save the model
    data_dir: Path to data
    test_speaker: target speaker (Example: 'se')
- AllSpeak_WJDOT_data_configuration: it contains a class to load data for AllSpeak_SI_model.py.
    Please modify this file to run the model on your own data.
- AllSpeak_WJDOT.py: it performs Speaker-Adaptation based on WJDOT. It takes three command-line arguments:
    saver_dir: Path to save the model
    data_dir: Path to data
    test_speaker: target speaker (Example: 'se')
- AllSpeak_WJDOT_data_configuration: it contains the loading functions for AllSpeak_WJDOT.py.
    Please modify this file to run the model on your own data.
- command_mapping.py: it contains the map from the command number to the command sentence in AllSpeak dataset.
- MSDA_WJDOT.py: contains the wjdot algorithm without and with early stopping (based on the sse or the weighted sum of the accuracies)
- ot_torch.py: contains the optimal transport functions (e.g., objective function of Eq. (8) in the paper)