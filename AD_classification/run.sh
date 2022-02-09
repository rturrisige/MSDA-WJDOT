: '
# rosanna.turrisi@edu.unige.it
# EXPERIMENT PIPELINE: from image processing to model training and evaluation.

# Codes:
- from_nii_to_npy converts MRI file to nii into numpy and associates a label (CN or AD)
  NECESSARY PARAMETERS: data_path
- img_processing_and_data_augmentation rescale the original images and perform data augmentation by zoom, shift, and
  rotation
  NECESSARY PARAMETERS: data_path
- training_and_evaluation performs the model training and tests it on the test dataset.
  NECESSARY PARAMETERS: data_path, nexp, augmentation, ndataset, n_conv_features, saver_path


# Parameters to set up:
- data_path is the path to the folder containing the ADNI data: must to be a string
- saver_path is the path to the folder in which the model and the results will be saved: must to be a string
- nexp is the number of experiment performing model classification: must to be an integer
- augmentation is the type augmented data to use in the model tranining: must be one of the following strings
  "0" corresponds to non-augmented data,
  "1" corresponds to zoom,
  "2" corresponds to shift,
  "3" corresponds to rotation,
  "123" applies separately zoom, shift, and rotation,
  "4" applied all transformations (zoom, shift,rotation) simultaneously.
- ndataset sets how many augmented datasets have to be used: it must be an integer
  it must be set to 1 for strategy A and B, whereas it has to be fixed at 3 for strategy C.
- n_conv is the number of convolutional layers

# In the following, the parameters are set up in oder to carry out the papiline for model 8 CL with strategy B.
 '
data_path='ADNI_toy'/
saver_path='Results/net_8CL/'


# Data preparation: image pre-processing and data augmentation
python from_nii_to_npy $data_path # converts files from nii to numpy and creates a txt file containing images size information
python img_processing_and_data_augmentation.py $data_path  # to apply image scaling and data augmentation

# Set parameters for training and testing:
augmentation="123"
ndataset=1
n_conv=8

# Run training and testing:
for nexp in {1..10}
do
  echo "Experiment" $nexp ": AD/CN binary classification - saved at" $saver_path
  python training_and_evaluation.py $data_path $nexp $augmentation $ndataset $n_conv $saver_path
done
