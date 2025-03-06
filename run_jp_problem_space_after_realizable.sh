
setting=problem_space_realizable2171_add_side_effect_fea
mkdir -p logs/$setting

family=("plankton" "mobisec" "adwo" "youmi" "cussul" "tencentprotect" "anydown" "leadbolt" "airpush" "kuguo")
devices=(0 1 2 3 4 5 6 7 0 1) # GPU device IDs



i=42
iter=40
malrate=0
benign_ratio=0.001
lambda=1e-3
for idx in ${!family[@]}; do
    nohup python problem_space_with_side_effect_fea.py -D \
                         --subset-family ${family[$idx]}         \
                         --clean-ratio 0.02              \
                         --max-iter $iter                   \
                         --num-of-train-batches 5        \
                         -d apg                          \
                         -c mlp                          \
                         --lambda-1 $lambda              \
                         --num-triggers 1                \
                         --benign-poison-ratio $benign_ratio      \
                         --mask-optim-step 30            \
                         --attack-succ-threshold 0.85    \
                         --mlp-hidden 1024               \
                         --mlp-lr 0.001                  \
                         --mlp-batch-size 128            \
                         --mlp-epochs 30                 \
                         --mlp-dropout 0.2               \
                         --mask-expand-type 0            \
                         --poison-mal-benign-rate $malrate   \
                         --subset-benign-rate 5          \
                         --remain-benign-rate 1          \
                         --random-state $i               \
                         --mask-size-upperbound 0        \
                         --convert-mask-to-binary 0      \
                         --mntd-half-training 0          \
                         --use-last-weight 1             \
                         --alter-retrain-full-training 0 \
                         --device ${devices[$idx]} > logs/$setting/${family[$idx]}-lambda-$lambda-$(date "+%m.%d-%H.%M.%S").log &
done
