#!/bin/sh

# First argument is the path to the list of images

python grad_cam.py -c [config file] -ckpt [chekcpoint]] --gpu -f $1 


# write this out for coarse, fine and bilateral