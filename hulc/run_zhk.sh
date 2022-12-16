#!/bin/sh
CUDA_VISIBLE_DEVICES=1,2  taskset -c 0-18 \
python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_D_D \
datamodule/datasets=vision_lang_shm \
~callbacks/rollout \
~callbacks/tsne_plot \
model/action_decoder=skill \
model.action_decoder.perceptual_emb_slice.0=0 \
model.action_decoder.out_features=18 \
model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR_2022-12-09_21-34-22' \
model/distribution=discrete \
loss.kl_beta=0.0005 \
loss.clip_auxiliary_loss_beta=0.003 \
model.action_decoder.beta=0.001
