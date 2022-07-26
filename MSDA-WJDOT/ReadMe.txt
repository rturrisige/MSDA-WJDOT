########################################### READ ME #############################################

The codes have been tested on Python 3.7.6. In order to run, the following Python modules       
are required:

- Numpy/SciPy
- PyLab
- Matplotlib
- PyTorch and dependencies from http://pytorch.org.
- POT python package 
- PyDub
- librosa


######################
# Reproduce plots   ##
######################


- visualization_2D_domain_shift.py: plots Figure 1 in the paper (illustration of MSDA-WJDOT on 2D simulated data - domain shift)
- visualization_3D_domain_shift.py: similarly to Figure 3 in the paper, it plots the accuracy obtained by exploiting the Exact OT and Bures and the recovered alpha weights for an increasing target rotation angle. 
- visualization_2sources_target_shift.py: illustration of MSDA-WJDOT on target shift problem. It plots Source and Target data (Figure 4: left), the reweighted sources and the found classifier (Figure 4 left-center) and the alpha coefficients (Figure 4: right).

#####################
# Simulations      ##
#####################

- ot_torch.py: contains the optimal transport functions (e.g., objective function of Eq. (8) in the paper)
- wjdot.py: contains the wjdot algorithm (Algorithm 1 in the paper) without and with early stopping (based on the sse or the weighted sum of the accuracies)

- simulated_data_domain_shift.py: generates 3D-Gaussian distributed source and target datasets with different angle rotations and applies the  MSDA-WJDOT algorithm to solve MSDA with domain shift.
- simulated_data_target_shift: generates 2D-Gaussian distributed source and target datasets with different proportions of classes and applies the MSDA-WJDOT algorithm to solve MSDA with target shift.

- simulated_data_domain_shift_bound.py: generates 3D-Gaussian distributed source and target datasets with different angle rotations and applies the  MSDA-WJDOT algorithm giving the optimal Alpha. This is used to compute the upper bound of the Lambda term in Theorem 1. Similarly, the upper bound is computed by using random and uniform Alpha weights. 

- simulated_data_target_shift: generates 2D-Gaussian distributed source and target datasets with different proportions of classes and applies the MSDA-WJDOT algorithm to solve MSDA with target shift.

- object_recognition.py: takes as input the datasets in the embedding, split in training, validation and testing. 
  Each of these is a list of matrices with dimension [number of samples, embedding dimension + 1], representing the concatenation of the embedding features and the labels.
  WJDOT is applied by leaving out one dataset at a time as target domain. 

- music_speech_processing.py: contains the functions to generate the noisy datasets and to extract the MFCCs from the audio files.
- multi_task_learning.py: contains the multi-task-learning (MTL) functions and model.
- music_speech_discrimination.py: runs the MTL training, extracts the feature embedding from target and source datasets and applies the wjdot algorithm. It takes 3 command-line arguments:
  1) the path of the folder containing the data; 2) an integer between 0 and 3 to choose the target domain; 3) the early stopping strategy that can be "acc" or "sse".
  
 
