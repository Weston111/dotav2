python demo/image_demo.py \
       --inputs images/ \
       --model configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py \
       --weights ../weights/lae_dino_swint_lae1m-28ca3a15.pth \
       --texts 'playground . road . tank . airplane . vehicle . bridge' -c \
       --palette random \
       --out-dir outputs \
       --pred-score-thr 0.4