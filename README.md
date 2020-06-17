# VAELLS
 This code accompanies the paper "Variational Autoencoder with Learned Latent Structure" - M. Connor, G. Canal, and C. Rozell

## Requirements
- torch
- scipy
- numpy
- cv2
- sklearn
- six
- matplotlib
- math
- glob
- logging

## Code structure

### Main VAELLS Function
`VAELLS.py` is the main file to run when training the VAELLS model. The argparser at the beginning of the file shows the possible parameters to select for a run. One notable parameter selection is `data_use` which specifies the dataset you'll be working with. The options are `concen_circle`,`swiss2D`,`rotDigits`, and `natDigits` as in the experiments in the VAELLS paper. Training outputs include network_...pt files which track the weights that are being trained and spreadInferenceTest files which save outputs that are useful when monitoring the performance of the model as training progresses.


### Auxillary functions
- `batch_TOVAE_train.py` - Used to run batches of parameter combination to aid in parameter selection
- `covNetModel.py` - Defines the convolutional networks used in the MNIST experiments
- `fullyConnectedModel.py` - Defines the fully connected networks used in the swiss roll and concentric circle experiments
- `test_metrics_MNIST_natDigit.py` - functions for computing log-likelihood, MSE, and ELBO for natural MNIST digit tests
- `test_metrics_MNIST_rotDigit.py` - functions for computing log-likelihood, MSE, and ELBO for rotated MNIST digit tests
- `TOVAE_computeMetrics.py` - code for loading a pre-trained model from rotated MNIST or natural MNIT tests and computing log-likelihood, MSE, and ELBO
- `trans_opt_objectives.py` - functions used to compute portions of the VAELLS transport operator objective
- `transOptModel.py` - defines the transport operator neural network layer
- `utils.py` - contains functions for reading in and generating data as well as visualizing outputs

## Guide to make plots
*Figure 2 and 3* (swiss roll experimental results):
 - To train VAELLS: run `VAELLS.py` with data_type: `swiss2D`
 - To make data for plots: run `createDataPlots_swissRoll2D.py`. Make sure to specify the checkpoint file associated with the trained model you're interested in viewing results for. 
 - To create plots: run `plotSwissRollOutputs.m`
 
*Figure 4, 7, 8, 9* (concentric circle experimental results):
 - To train VAELLS: run `VAELLS.py` with data_type: `concen_circle`
 - To make data for plots: run `createDataPlots_concen_circle.py`. Make sure to specify the checkpoint file associated with the trained model you're interested in viewing results for. 
 - To create plots: run `plotCircleOutputs.m` 

*Figure 5* (rotated MNIST experimental results)
 - To train VAELLS: run `VAELLS.py` with data_type: `rotDigits`
 - To make data for plots: run `genTransOptSeq_rotDigits.py`. Make sure to specify the checkpoint file associated with the trained model you're interested in viewing results for. 
 - To create plots: run `plotTransOptImgOrbits_TOVAE_rotDigits.m` 

*Figure 6* (natural MNIST experimental results)
 - To train VAELLS: run `VAELLS.py` with data_type: `natDigits`
 - To make data for plots: run `genTransOptSeq_natDigits.py`. Make sure to specify the checkpoint file associated with the trained model you're interested in viewing results for. 
 - To create plots: run `plotTransOptImgOrbits_TOVAE_natDigits.m` 
 - To compute metrics for table: run `TOVAE_computeMetrics.py`
