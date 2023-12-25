#!/bin/bash

bash ./grad_cam.sh grad_cam_images/distribution_images_ntbt-bicamntbt.txt

bash ./grad_cam.sh grad_cam_images/distribution_images_ntbt-bicamnfbt.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_nfbt-bicamnfbt.txt
bash ./grad_cam.sh grad_cam_images_images_ntbf-bicamnfbt.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_nfbf-bicamnfbt.txt

bash ./grad_cam.sh grad_cam_images/distribution_images_ntbt-bicamntbf.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_nfbt-bicamntbf.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_ntbf-bicamntbf.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_nfbf-bicamntbf.txt

bash ./grad_cam.sh grad_cam_images/distribution_images_nfbt-bicamnfbf.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_ntbf-bicamnfbf.txt
bash ./grad_cam.sh grad_cam_images/distribution_images_nfbf-bicamnfbf.txt