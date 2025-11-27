# Training script for DeepLabV3+ with 100 labeled base images
# Expected improvement: 65.12% → 71-75% mIoU (+6-10%)

C:\Users\IoT-C504-03\miniconda3\envs\als4gan_env\python.exe tools/train_s4gan_salak_v3.py `
    --data-root "C:/_albert/s4GAN/patchify/temp_patches" `
    --class-mapping "C:/_albert/ALS4GAN/class_mapping.csv" `
    --labeled-files "labeled_files_100.txt" `
    --checkpoint-dir "C:/_albert/ALS4GAN/checkpoints_v3_improved_2411" `
    --num-classes 7 `
    --batch-size 4 `
    --num-steps 75000 `
    --save-pred-every 5000 `
    --eval-every 2500 `
    --learning-rate 0.001 `
    --st-loss-weight 1.5 `
    --early-stop-patience 40 `
    --backbone resnet50 `
    --output-stride 16 `
    --num-heads 3 `
    --wandb-project "als4gan-salak" `
    --wandb-run-name "v3_100labeled"
