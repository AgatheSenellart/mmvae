#!/bin/bash


python3 src/bivae/main.py --config-path src/configs_experiments/circles/jmvae_nf.json
python3 src/bivae/main.py --config-path src/configs_experiments/circles/jmvae.json
python3 src/bivae/main.py --config-path src/configs_experiments/circles/jmvae_nf_dcca.json
# python3 src/bivae/main.py --config-path src/configs_experiments/circles/jmvae_nf_recon.json
# python3 src/bivae/main.py --config-path src/configs_experiments/circles/jmvae_nf_dcca_recon.json

python3 src/bivae/toy_plot.py