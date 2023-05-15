
# -------- bilateral --------

# train 5 specialized hemispheres, both fine and coarse
# and then train the bilateral model using them, each time with a different seed
python train_system.py --arch vgg11 --num_seeds 5 --epochs 1 1 1 --uni_base_config configs/config_unilateral_specialize.yaml --bi_base_config configs/config_bilateral_specialized.yaml



# -------- baseline: non specialised bilateral network   --------
python trainer.py --config configs/config_bilateral_nspecialized.yaml



# -------- baseline: ensemble   --------

# train 5 non-specialized hemispheres (seeds in the config)
python trainer.py --config configs/config_unilateral_nspecialized.yaml

# train ensemble    --------> manually set the names of the checkpoints then run this:
# python trainer.py --config configs/config_ensemble.yaml

