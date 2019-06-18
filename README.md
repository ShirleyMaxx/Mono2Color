#---------------------------------------------------------------
# Project
# From monochromatic to color image
# 2019-06-18
# Author: Molly & Shirley
# --------------------------------------------------------------

1. This folder contains two directories: cyclegan + dataset

2. Download cifar_images.zip, put it under 'dataset' dir and unzip it.

3. Directory 'cyclegan' contains the code.
    $ cyclegan
        - train.py
        - test.py
        - \_init_paths.py
        - experiments
            - default.yaml
        - lib
            - models
                - cyclegan.py
            - config.py
            - dataset.py
            - function.py
            - utils.py

4. Train
    - cd to 'cyclegan' dir, and run
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cfg experiments/default.yaml
    - then under 'cyclegan' dir, there will be two dirs added.
        - log
            - run 
            $ tensorboard --logdir ./
            will see the recorded loss in the tensorboard
        - black2rgb 
            - contains the output:
                - the generated color images
                - checkpoint.pth.tar
                - training log
    - All the hyper-parameters are in experiments/default.yaml, change them to train another networks in an easy way.

5. Test
    - after training, keep to 'cyclegan' dir, and run
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --cfg experiments/default.yaml
    - the code will load the latest checkpoint and test it on the testing images.
    - As memory limit, we leave out the checkpoint.pth.tar in the submitted zip.

6. The generated tesing images are under 'project\cyclegan\black2rgb\cyclegan\default\images'
