# Depth Map Prediction Using U-Net
## Data
http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
## Refs
- http://www.cs.nyu.edu/~deigen/depth
- Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
- Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
## UNet 
1. The Original Unet Paper
    https://arxiv.org/pdf/1505.04597.pdf
2. Our implementation base
    https://github.com/milesial/Pytorch-UNet

## use cases
python3.6 train.py -e 30 -b 5 -g --dir 'e30_b5_half_eps5_lamb1'
python3.6 predict.py --model /data/chc631/project/data/checkpoints/e30_b5_half_eps5_lamb1/CP10.pth -i red_couch.jpg -o red_couch_e30_b5_half_eps5_lamb1_CP10.mat
