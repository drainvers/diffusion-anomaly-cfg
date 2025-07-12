# Integrating Deblurring Techniques with Diffusion Models to Localize Thoracic Abnormalities in Chest X-rays

This is a fork of the PyTorch implementation of the code for ["Diffusion Models for Medical Anomaly Detection"](https://arxiv.org/abs/2203.04306), with modifications for classifier-free guidance. The codebase is derived from [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

## Data

We trained the model on the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) and evaluated on an externally provided anonymized test set of 11 images with known signs of pleural effusion. A small example of the directory structure for the dataset can be found in the folder *data*. To train or evaluate on the desired dataset, set `--dataset chexpert` and specify the directory. An additional `ChexpertDataset` class has been implemented to allow working with the CheXpert-v1.0 directory structure directly, simply comment the `load_data()` call and use the `ChexpertDataset` class insead. If using this class, please specify the root folder of the dataset, and set the train-test flag in the class initialization. If using `load_data()`, directly set the path to the `training` folder for training or the `testing` folder for evaluation.

We include a preprocessing script to downsize images to 256x256. Annotations are kept mostly as-is with changes to the extension in the image path to match our downsized image format. Label filtering is performed in the ChexpertDataset class for splitting into training and testing sets. The separation of data into "healthy" and "diseased" images is done with Pandas, where healthy data is defined as `df["No Finding"] == 1` and diseased as `df["Pleural Effusion"] == 1`. We sample 16,000 images for each class.

## Usage

We set the flags as follows:
```
MODEL_FLAGS="--unet_version v1 --image_size 256 --in_channels 1 --num_channels 128 --num_classes 2 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --p_uncond 0.1 --dropout 0.1"
SAMPLE_FLAGS="--batch_size 1 --num_samples 11 --timestep_respacing ddim1000 --use_ddim True"
```
To train the diffusion model, run
```
python scripts/cfg_image_train.py --data_dir --data_dir path_to_traindata --dataset chexpert $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
The model will be saved in the *results* folder (can be changed through the `--result_dir` argument).

For image-to-image translation to a pseudo-healthy subject on the test set, run
```
python scripts/cfg_image_sample.py --data_dir path_to_testdata --model_path ./results/model.pt --dataset brats_or_chexpert --guidance_scale 4.0 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 
```
A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).

## Deblurring component

Our deblurring component follows the implementation in this [repository](https://github.com/yuanzhi-zhu/DiffPIR), with some modifications to allow use of our own model prior. We construct an unconditional model with the same hyperparameters as our main model, and train on the VinDR-CXR dataset (train and test merged). To compensate for computational limitations, we run this process on the data to generate a deblurred version of the CheXpert dataset before training and evaluating on the models used in the following comparison. We set blur type to motion, with a kernel size of 10, and a low noise level of 0.5.

## Comparing Methods

### FixedPoint-GAN

We follow the implementation given in this [repository](https://github.com/mahfuzmohammad/Fixed-Point-GAN). We choose the following hyperparameters:
- λ<sub>cls</sub>=1
- λ<sub>gp</sub>=λ<sub>id</sub>=λ<sub>rec</sub>=10
- g_conv_dim=128
- g_repeat_num=14
and train our model for 50000 iterations. The batch size is set to 4, and the learning rate to 10<sup>-4</sup>.

### DDIM with classifier guidance

We follow the implementation given in this [repository](https://github.com/JuliaWolleb/diffusion-anomaly) and train the model for 50000 iterations. We use the same hyperparameters as in this repository for our classifier-free guidance version, with the exception of the classifier hyperparameters which are unused. The batch size is set to 4, and the learning rate to 10<sup>-4</sup>.