##########################################################################
########################  Train MLP classifier  ##########################

###### limit to 10000 features
###### full training, 30 epochs, batch_size 128
mkdir -p logs/apg/mlp/
i=42
j=0
fea=10000
#nohup 
python -u backdoor_general.py -D               \
                        -R CLASSIFIER        \
                        -d apg               \
                        -c mlp               \
                        --mlp-retrain 0      \
                        --mlp-hidden 1024    \
                        --mlp-lr 0.001       \
                        --mlp-batch-size 128  \
                        --mlp-epochs 30      \
                        --mlp-dropout 0.2    \
                        --n-features $fea   \
                        --random-state $i    \
                        --mntd-half-training 0    \
                        --device $j           
                        #\
                        #> logs/apg/mlp/mlp-full-$fea-feature-hidden1024-epoch30-batch128-half0-r$i-$(date "+%m.%d-%H.%M.%S").log &



############### imperfect attacker, do not know the MLP clean model architecture, only knows 10000 features
# mkdir -p logs/apg/mlp/
# i=42
# j=6
# nohup python -u backdoor_general.py -D               \
#                         -R CLASSIFIER        \
#                         -d apg               \
#                         -c mlp               \
#                         --mlp-retrain 0      \
#                         --mlp-hidden 32    \
#                         --mlp-lr 0.001       \
#                         --mlp-batch-size 128  \
#                         --mlp-epochs 30      \
#                         --mlp-dropout 0.2    \
#                         --n-features 10000   \
#                         --random-state $i    \
#                         --mntd-half-training 0    \
#                         --device $j           \
#                         > logs/apg/mlp/mlp-full-10000feature-hidden32-epoch30-batch128-half0-r$i-$(date "+%m.%d-%H.%M.%S").log &




# ############## half training for MNTD benign target models, 20 epochs, batch_size 128, need to change j to avoid GPU memory error
# # mkdir -p logs/apg/mlp/anydown
# mkdir -p logs/apg/mlp/half_train
# # NOTE: test one experiment first
# # for i in {0..0}; do
# #     j=0


# # --subset-family anydown \
# # NOTE: make sure the first one is correct, then run others
# for i in {1..31}; do
#     j=$(($i / 8)) # or just use i instead of $i, $(()) does an arithmetic evaluation, each GPU node run 8 clean model training
#     e=20
#     nohup python -u backdoor_general.py -D    \
#                          -R CLASSIFIER        \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 128  \
#                          --mlp-epochs $e      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --random-state $i    \
#                          --mntd-half-training 1    \
#                          --device $j           \
#                          > logs/apg/mlp/half_train/mlp-full-10000feature-hidden1024-epoch$e-batch128-half1-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done




#####################################################################################
################### full training, repeat 5 times, for STRIP eval ###################

# part=top
# size=10
# mkdir -p logs/baseline/mlp_full_train/$part-$size/
# # NOTE: test one experiment first
# for i in {42..42}; do
#     device=0
# # for i in {2..5}; do
#     # device=2
#     # device=$(($i))
# # for i in {16..31}; do
# #     device=$(($i / 2-8)) # or just use i instead of $i, $(()) does an arithmetic evaluation, each GPU node run 8 training
#     e=30
#     nohup python -u backdoor_general.py -D    \
#                          -R CLASSIFIER        \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 128  \
#                          --mlp-epochs $e      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --random-state $i    \
#                          --mntd-half-training 0     \
#                          --select-benign-features $part \
#                          --trojan-size $size        \
#                          --backdoor                 \
#                          --device $device           \
#                          > logs/baseline/mlp_full_train/$part-$size/backdoor-e$e-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done





##################################################################################################################
################### full training, repeat 1 time, for Activation Clustering eval , clean model ###################


# mkdir -p logs/apg/mlp_full_train_remove_subset_and_dup/
# # NOTE: test one experiment first
# # family=mobisec
# # family=leadbolt
# family=tencentprotect
# for i in {42..42}; do
#     device=7
#     e=30
#     nohup python -u backdoor_general.py -D    \
#                          -R CLASSIFIER        \
#                          -d apg               \
#                          -c mlp               \
#                          --subset-family $family \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 128  \
#                          --mlp-epochs $e      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --random-state $i    \
#                          --mntd-half-training 0     \
#                          --device $device           \
#                          > logs/apg/mlp_full_train_remove_subset_and_dup/remove-$family-clean-model-e$e-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done







# half training for MNTD baseline backdoor target models, 20 epochs, batch_size 128, need to change j to avoid GPU memory error
# mkdir -p logs/apg/mlp/anydown



# --middle-N-benign 1000     \


# # part=bottom
# part=top
# size=5
# mkdir -p logs/baseline/mlp_half_train/$part-$size/
# # NOTE: test one experiment first
# # for i in {0..0}; do
# #     device=0
# for i in {1..15}; do
#     device=$(($i / 2))
# # for i in {16..31}; do
# #     device=$(($i / 2-8)) # or just use i instead of $i, $(()) does an arithmetic evaluation, each GPU node run 8 training
#     e=20
#     nohup python -u backdoor_general.py -D    \
#                          -R CLASSIFIER        \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 128  \
#                          --mlp-epochs $e      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --random-state $i    \
#                          --mntd-half-training 1     \
#                          --select-benign-features $part \
#                          --trojan-size $size        \
#                          --backdoor                 \
#                          --device $device           \
#                          > logs/baseline/mlp_half_train/$part-$size/backdoor-e$e-r$i-$(date "+%m.%d-%H.%M.%S").log &
# done

###### only use 10% of apg dataset, use full features
# mkdir -p logs/apg-10/
# nohup python backdoor_general.py -D                  \
#                          -R CLASSIFIER       \
#                          -d apg-10           \
#                          -c mlp              \
#                          > logs/apg-10/mlp-full-$(date "+%m.%d-%H.%M.%S").log &
##########################################################################



# hasn't run the MLP for the new evaluation method

##########################################################################
#################  Backdoor attack (feature space) #######################
############  TOP  ############
# mkdir -p logs/apg/
# nohup python backdoor_general.py -D               \
#                          -R ATTACK            \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 64  \
#                          --mlp-epochs 30      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --select-benign-features top \
#                          --backdoor > logs/apg/mlp/backdoor-mlp-full-10000feature-top-$(date "+%m.%d-%H.%M.%S").log &


# ############  MIDDLE  ############
# mkdir -p logs/apg/
# nohup python backdoor_general.py -D               \
#                          -R ATTACK            \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 64  \
#                          --mlp-epochs 30      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --middle-N-benign 1000  \
#                          --device 5               \
#                          --backdoor > logs/apg/mlp/backdoor-mlp-full-10000feature-middle1000-$(date "+%m.%d-%H.%M.%S").log &

# ############  BOTTOM  ############
# mkdir -p logs/apg/
# nohup python backdoor_general.py -D               \
#                          -R ATTACK            \
#                          -d apg               \
#                          -c mlp               \
#                          --mlp-retrain 0      \
#                          --mlp-hidden 1024    \
#                          --mlp-lr 0.001       \
#                          --mlp-batch-size 64  \
#                          --mlp-epochs 30      \
#                          --mlp-dropout 0.2    \
#                          --n-features 10000   \
#                          --select-benign-features bottom \
#                          --backdoor > logs/apg/mlp/backdoor-mlp-full-10000feature-bottom-$(date "+%m.%d-%H.%M.%S").log &
##########################################################################

