# unet-segmentation-kaggle
Part of Kaggle challenge https://www.kaggle.com/c/severstal-steel-defect-detection

Contains code which were part of the solution. Personal best leaderboard score was `0.85521`, winning leaderboard was `0.90883` (Dice co-efficient)


### Solution details

Best result was achieved with a UNet with the downscaler using a VGG16 with pretrained imagenet-weights. Compared to a full image Unet, a tiled approach where the image is split into 4 divisions and trained on improved scores by 4%. Playing with the thresholds further improved the score by 0.6%. A sample weighted categorical cross entropy loss was used (a sample weighted sparse categorical cross entropy loss can also be used) to deal with class imbalance. Training was only done with data containing atleast one of the defect classes. a `StratifiedSfuffleSplit` was was to create 3-5 fold cross validation data and the final model was a ensemble of 3 such models. Also tried a classifier that would filter images having defects vs non-defect images as a first step before the UNet. In the end VGG16 being parameter heavy was a problem. ToDo: explore with lighter classifiers esp `Efficienets` 
