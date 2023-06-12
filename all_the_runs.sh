
echo ----------------------------------------------------------------
echo -------- baseline: non-specialised bilateral network   ---------
echo ----------------------------------------------------------------
python trainer.py --config configs/config_bilateral_nspecialized.yaml


echo ----------------------------------------------------------------
echo -------- baseline: ensemble   --------
echo ----------------------------------------------------------------

# # train 5 non-specialized hemispheres (seeds in the config)
python trainer.py --config configs/config_unilateral_nspecialized.yaml

# echo # train ensemble    --------> manually set the names of the checkpoints then run this:
# # python trainer.py --config configs/config_ensemble.yaml

echo ----------------------------------------------------------------
echo -------- bilateral --------
echo ----------------------------------------------------------------

# train 5 specialized hemispheres, both fine and coarse
# and then train the bilateral model using them, each time with a different seed
python train_system.py --arch resnet9 --num_seeds 5 --epochs 180 180 180 --uni_base_config configs/config_unilateral_specialize.yaml --bi_base_config configs/config_bilateral_specialized.yaml

# if you already have the checkpoints (in trained_models.yaml) you can just run this:
# python train_system.py --arch vgg11 --num_seeds 5 --epochs 1 1 1 --bi_base_config configs/config_bilateral_specialized.yaml --trained_models configs/trained_models.yaml 


