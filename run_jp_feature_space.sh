##########################################################################################################################
################################################# Default Setting, Table 1 and 15 ##########################################
##########################################################################################################################

# Table 1
family=("plankton" "mobisec" "adwo" "youmi" "cussul" "tencentprotect" "anydown" "leadbolt" "airpush" "kuguo")
devices=(0 1 2 3 4 5 6 7 0 1) # GPU device IDs

# Table 15:
# family=("dowgin" "artemis" "jiagu" "revmob" "genpua" "feiwo" "smspay" "eldorado" "igexin" "deng" "baiduprotect")
# devices=(0 1 2 3 4 5 6 7 0 1 2)


i=42 # random state
batches=5
subset_benign_rate=5 # lambda_1
remain_benign_rate=1 # lambda_2
poison_ratio=0.001   # 100 benign
alterfull=0 # train the alternate optimization model with X_train_batch + X_poison
malrate=0 # 0 mal
half_train=0

mode=feature-space-attack
mkdir -p logs/$mode/

clean_ratio=0.02
step=30
iter=40
last_weight=1
for idx in ${!family[@]}; do
    nohup python -u subset_backdoor_main.py                         \
                         -d apg                                     \
                         -c mlp                                     \
                         --setting $mode                            \
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
                         --device ${devices[$idx]} > logs/$mode/${family[$idx]}-main-subset$subset_benign_rate-remain$remain_benign_rate-malrate$malrate-poison$poison_ratio-thre0.85-numbatch$batches-mask-real_lambda-1e-3_step$step-clean$clean_ratio-uselast$last_weight-alterfull$alterfull-bs128_r$i-$(date "+%m.%d-%H.%M.%S").log &
done





# ##########################################################################################################################
# ################################################# for subset backdoor, limited data #################################
# family=("leadbolt")  # run one family first to generate the clean model
# devices=(7)
# # family=("mobisec" "plankton" "airpush" "tencentprotect")
# # family=("mobisec" "leadbolt" "tencentprotect")
# # family=("kuguo" "jiagu" "igexin" "baiduprotect")
# # devices=(0 1 2 3)
# # devices=(4 5 6 7)
# # family=("cussul")
# # devices=(4)
# i=42
# # iter=100 # iter=40
# iter=40
# # iter=20
# batches=5
# subset_benign_rate=5
# remain_benign_rate=1
# poison_ratio=0.001   # 100 benign

# alterfull=0
# malrate=0 # 0 mal
# half_train=0
# mode=limited_training_data_0.1
# clean_ratio=0.2 # 11262022 for leadbolt limited 0.1
# ## clean_ratio=0.2 # for limited_data_0.1 # TODO: maybe change it to 0.1 is better?
# # clean_ratio=0.1 # rerun to keep the same as other ratios, this is the setting for all other experiments
# # mode=limited_training_data_0.2
# # clean_ratio=0.1 # for limited_data_0.2
# # mode=limited_training_data_0.3
# # clean_ratio=0.1 # for limited_data_0.3
# # mode=limited_training_data_0.5
# # clean_ratio=0.02 # for limited_data_0.5
# mkdir -p logs/$mode/

# # clean_ratio=0.02

# step=30
# last_weight=1
# for idx in ${!family[@]}; do
#     nohup python -u subset_backdoor_main_remove_duplicate_vectors_limited_data.py -D                \
#                          -d apg                                     \
#                          -c mlp                                     \
#                          --setting $mode                            \
#                          --lambda-1 1e-3                            \
#                          --num-triggers 1                           \
#                          --benign-poison-ratio $poison_ratio        \
#                          --clean-ratio $clean_ratio                 \
#                          --max-iter $iter                           \
#                          --num-of-train-batches 5                   \
#                          --mask-optim-step $step                    \
#                          --attack-succ-threshold 0.85               \
#                          --mlp-hidden 1024                          \
#                          --mlp-lr 0.001                             \
#                          --mlp-batch-size 128                       \
#                          --mlp-epochs 30                            \
#                          --mlp-dropout 0.2                          \
#                          --mask-expand-type 0                       \
#                          --poison-mal-benign-rate $malrate          \
#                          --subset-benign-rate $subset_benign_rate   \
#                          --remain-benign-rate $remain_benign_rate   \
#                          --random-state $i                          \
#                          --subset-family ${family[$idx]}            \
#                          --mask-size-upperbound 0                   \
#                          --convert-mask-to-binary 0                 \
#                          --mntd-half-training $half_train           \
#                          --use-last-weight $last_weight             \
#                          --alter-retrain-full-training $alterfull   \
#                          --device ${devices[$idx]} > logs/$mode/${family[$idx]}-poison-$poiso_ratio-halftrain-$half_train-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done





##########################################################################################################################
################################################# for subset backdoor, imperfect attacker #################################
# # family=("leadbolt")  ########### WARNING: run one family first to generate the clean model
# # devices=(0)
# # family=("mobisec" "plankton" "tencentprotect")
# # devices=(1 2 3)
# family=("kuguo")
# devices=(1)
# # hidden=2048
# hidden=32
# i=42
# # iter=40
# iter=100
# batches=5
# subset_benign_rate=5
# remain_benign_rate=1
# poison_ratio=0.001   # 100 benign

# alterfull=0
# malrate=0 # 0 mal
# half_train=0
# mode=imperfect_attacker_hidden_$hidden
# mkdir -p logs/$mode/

# clean_ratio=0.02
# step=30
# last_weight=1
# for idx in ${!family[@]}; do
#     nohup python -u subset_backdoor_main_remove_duplicate_vectors_imperfect_attacker.py -D                \
#                          -d apg                                     \
#                          -c mlp                                     \
#                          --setting $mode                            \
#                          --lambda-1 1e-3                            \
#                          --num-triggers 1                           \
#                          --benign-poison-ratio $poison_ratio        \
#                          --clean-ratio $clean_ratio                 \
#                          --max-iter $iter                           \
#                          --num-of-train-batches 5                   \
#                          --mask-optim-step $step                    \
#                          --attack-succ-threshold 0.85               \
#                          --mlp-hidden 1024                          \
#                          --mlp-lr 0.001                             \
#                          --mlp-batch-size 128                       \
#                          --mlp-epochs 30                            \
#                          --mlp-dropout 0.2                          \
#                          --mask-expand-type 0                       \
#                          --poison-mal-benign-rate $malrate          \
#                          --subset-benign-rate $subset_benign_rate   \
#                          --remain-benign-rate $remain_benign_rate   \
#                          --random-state $i                          \
#                          --subset-family ${family[$idx]}            \
#                          --mask-size-upperbound 0                   \
#                          --convert-mask-to-binary 0                 \
#                          --mntd-half-training $half_train           \
#                          --use-last-weight $last_weight             \
#                          --alter-retrain-full-training $alterfull   \
#                          --device ${devices[$idx]} > logs/$mode/${family[$idx]}-hidden$hidden-poison-$poiso_ratio-halftrain-$half_train-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done



##########################################################################################################################
################################################# for subset backdoor, allow subset in training set #################################
# # family=("leadbolt")
# # devices=(3)
# # family=("mobisec" "plankton" "airpush" "leadbolt")
# # family=("kuguo" "jiagu" "igexin" "baiduprotect")
# # family=("smspay" "tencentprotect" "cussul")
# # devices=(0 1 2 3)
# # family=("leadbolt" "smspay" "tencentprotect" "cussul")
# # devices=(3 4 5 6)
# # devices=(5 6 7)
# # family=("anydown")
# # devices=(1)
# family=("kuguo")
# devices=(5)
# i=42
# iter=40
# batches=5
# subset_benign_rate=5
# remain_benign_rate=1
# poison_ratio=0.001   # 100 benign
# # poison_ratio=0.05

# alterfull=0
# malrate=0 # 0 mal
# half_train=0
# # half_train=1
# mode=allow-subset-in-train-halftrain-$half_train
# mkdir -p logs/$mode/

# clean_ratio=0.02
# step=30
# last_weight=1
# for idx in ${!family[@]}; do
#     nohup python -u subset_backdoor_main_remove_duplicate_vectors_allow_subset_in_train.py -D                \
#                          -d apg                                     \
#                          -c mlp                                     \
#                          --setting $mode                            \
#                          --lambda-1 1e-3                            \
#                          --num-triggers 1                           \
#                          --benign-poison-ratio $poison_ratio        \
#                          --clean-ratio $clean_ratio                 \
#                          --max-iter $iter                           \
#                          --num-of-train-batches 5                   \
#                          --mask-optim-step $step                    \
#                          --attack-succ-threshold 0.85               \
#                          --mlp-hidden 1024                          \
#                          --mlp-lr 0.001                             \
#                          --mlp-batch-size 128                       \
#                          --mlp-epochs 30                            \
#                          --mlp-dropout 0.2                          \
#                          --mask-expand-type 0                       \
#                          --poison-mal-benign-rate $malrate          \
#                          --subset-benign-rate $subset_benign_rate   \
#                          --remain-benign-rate $remain_benign_rate   \
#                          --random-state $i                          \
#                          --subset-family ${family[$idx]}            \
#                          --mask-size-upperbound 0                   \
#                          --convert-mask-to-binary 0                 \
#                          --mntd-half-training $half_train           \
#                          --use-last-weight $last_weight             \
#                          --alter-retrain-full-training $alterfull   \
#                          --device ${devices[$idx]} > logs/$mode/${family[$idx]}-poison-$poiso_ratio-halftrain-$half_train-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done
