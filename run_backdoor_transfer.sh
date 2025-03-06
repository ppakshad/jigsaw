family=mobisec
# family=leadbolt
# family=tencentprotect
# poison_ratio=0.005
poison_ratio=0.1

limited_data=0.1
# limited_data=0.3
device=0

mkdir -p logs/backdoor_transfer
i=42
CUDA_VISIBLE_DEVICES=$device nohup python -u backdoor_transfer_attacker_limited_data_target_svm.py -D          \
                        --space feature_space        \
                        --limited-data $limited_data \
                        -R ATTACK           \
                        -d apg              \
                        -c SVM              \
                        --n-features 10000  \
                        --svm-c 1           \
                        --svm-iter 10000     \
                        --subset-family $family        \
                        --benign-poison-ratio $poison_ratio      \
                        --poison-mal-benign-rate 0   \
                        --random-state $i             \
                        --device $device                    \
                        --backdoor > logs/backdoor_transfer/$family-$limited_data-limited-training-svm-full-10000feature-poison$poison_ratio-c1-iter10000-$(date "+%m.%d-%H.%M.%S").log &
