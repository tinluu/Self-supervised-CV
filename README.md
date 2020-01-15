# Self-supervised-CV
Class project (Fall 19 CV)
 
Please Notice: all the codes are inside code folders.

The Vae folder is the one method that we tried but not chosed.

1. First, make sure the torchvision's STL-10 dataset is in the ../ path. If not, just open the data_utils.py and set all the download arguments to True.
2. If you want to train the model, just run
`python train.py --gpu 0 --train 1`
Then you can train the model from the beginning.
If you want to train the model based on the pretrained weights. Run:
`python train.py --gpu 0 --train 1 --resume <path to the weights>`
3. To validate the model only:
`python train.py --gpu 0 --train 0 --resume <path to the weights>`
