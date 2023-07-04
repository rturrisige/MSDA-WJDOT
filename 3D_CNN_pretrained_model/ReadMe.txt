		3D-CNN pre-trained model for brain T1-weighted MRI
				
				Rosanna Turrisi


The model has been pre-trained on ADNI dataset, performing AD diagnosis based on brain T1w MRI.
This project is aimed for contributing in transfer learning in the healthcare domain.
The model can be imported from AD_pretrained_utilities.CNN and the weights are provided as "AD_pretrained_weights.pt". 
More information about the model pre-training can be found in AD_pretraining_info.txt.

Two implemented codes are here provided:

1) INFERENCE

It aims at using the pre-trained model for external validation. 
An example of how to run inference can be found in "run_inference_AD_pretrained.sh".

2) FEATURE EXTRACTOR

It aims at using the pre-trained model to extract features and create DL-based MRI embeddings.
An example of how to perform it can be found in "extract_embeddings_AD_pretrained.sh".


