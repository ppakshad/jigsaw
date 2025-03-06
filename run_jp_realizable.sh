realizable=1

family=("plankton" "mobisec" "adwo" "youmi" "cussul" "tencentprotect" "anydown" "leadbolt" "airpush" "kuguo")
devices=(0 1 2 3 4 5 6 7 0 1) # GPU device IDs

i=42
batches=5
subset_benign_rate=5
remain_benign_rate=1

poison_ratio=0.001   # 100 benign
alterfull=0
malrate=0
half_train=0


mode=realizable-2171
mkdir -p logs/$mode/$(date "+%m%d")/


clean_ratio=0.02
step=30
iter=40
last_weight=1
for idx in ${!family[@]}; do
    nohup python -u subset_backdoor_main.py -D                \
                         -d apg                                     \
                         -c mlp                                     \
                         --setting $mode                           \
                         --lambda-1 1e-3                            \
                         --num-triggers 1                           \
                         --benign-poison-ratio $poison_ratio        \
                         --clean-ratio $clean_ratio                 \
                         --max-iter $iter                           \
                         --num-of-train-batches $batches            \
                         --mask-optim-step $step                    \
                         --attack-succ-threshold 0.85               \
                         --mlp-hidden 1024                          \
                         --mlp-lr 0.001                             \
                         --mlp-batch-size 128                       \
                         --mlp-epochs 30                            \
                         --mlp-dropout 0.2                          \
                         --mask-expand-type 0                       \
                         --poison-mal-benign-rate $malrate          \
                         --subset-benign-rate $subset_benign_rate   \
                         --remain-benign-rate $remain_benign_rate   \
                         --random-state $i                          \
                         --subset-family ${family[$idx]}            \
                         --mask-size-upperbound 0                   \
                         --convert-mask-to-binary 0                 \
                         --mntd-half-training $half_train           \
                         --use-last-weight $last_weight             \
                         --alter-retrain-full-training $alterfull   \
                         --realizable-only $realizable              \
                         --device ${devices[$idx]} > logs/$mode/$(date "+%m%d")/${family[$idx]}-main-subset$subset_benign_rate-remain$remain_benign_rate-malrate$malrate-poison$poison_ratio-thre0.85-numbatch$batches-mask-real_lambda-1e-3_step$step-clean$clean_ratio-uselast$last_weight-alterfull$alterfull-bs128_r$i-$(date "+%m.%d-%H.%M.%S").log &
done
