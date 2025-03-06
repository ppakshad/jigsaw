#############################################################################################
#############################################################################################
#############################################################################################
################### step 1: generate benign (clean) shadow models
## WARNING: before running this, make sure svm-f10000.p (for SVM and MLP) exists
# i.e., execute ./run_backdoor_svm.sh first

#### MLP, use ~100% CPU,  6.3% CPU memory and 1117 MB GPU memory
mkdir -p logs/mntd/MLP/shadow_train/  &&
CUDA_VISIBLE_DEVICES=0 nohup python -u mntd_train_basic_benign.py \
                --task apg \
                --clf MLP  \
                > logs/mntd/MLP/shadow_train/benign-$(date "+%m.%d-%H.%M.%S").log &



#############################################################################################
#############################################################################################
#############################################################################################
################### step 2: generate poisoned (backdoored) shadow models with jumbo learning

####### implementation 1: (for subset attack, feature space) randomly select features from 10000 features as trojan
min=5
max=100
CUDA_VISIBLE_DEVICES=0 nohup python -u mntd_train_basic_jumbo.py \
                --task apg       \
                --clf MLP        \
                --troj_type M    \
                --min_size $min  \
                --max_size $max > logs/mntd/MLP/shadow_train/jumbo-2048-target-y-all-0-pattern-all-1-trojM-random-10000-size-$min-$max-$(date "+%m.%d-%H.%M.%S").log &


######## implementation 2: (for baseline attack) random select top 5-100 benign features as trojan
min=5
max=100
CUDA_VISIBLE_DEVICES=0 nohup python -u mntd_train_basic_jumbo.py \
                --task apg  \
                --clf MLP   \
                --troj_type Top-benign-jumbo  \
                --min_size $min               \
                --max_size $max > logs/mntd/MLP/shadow_train/jumbo-2048-top-benign-$min-$max-$(date "+%m.%d-%H.%M.%S").log &


######## implementation 3: (for problem space attack) random select top 5-100 features from those 2171 realizable features as trojan, space is much smaller
min=5
max=100
CUDA_VISIBLE_DEVICES=0 nohup python -u mntd_train_basic_jumbo.py \
                --task apg       \
                --clf MLP        \
                --troj_type Subset-2171-jumbo    \
                --min_size $min  \
                --max_size $max > logs/mntd/MLP/shadow_train/jumbo-2048-trojSubset-2171-target-y-all-0-pattern-all-1-size-$min-$max-$(date "+%m.%d-%H.%M.%S").log &






#############################################################################################
#############################################################################################
#############################################################################################
########### step 3: train target benign models with Pytorch  ################################

mkdir -p logs/mntd/MLP/target/benign/
family=mobisec
# family=leadbolt
# family=tencentprotect
CUDA_VISIBLE_DEVICES=0 nohup python -u mntd_train_target_benign.py \
                --task apg \
                --clf MLP  \
                --subset_family $family \
                > logs/mntd/MLP/target/benign/remove-$family-benign-target-256-$(date "+%m.%d-%H.%M.%S").log &



#############################################################################################
#############################################################################################
#############################################################################################
########### step 4: train target backdoor models with Pytorch (baseline and subset) #########

#################         baseline target backdoor       #################
##### baseline: generate backdoored target model using top benign features, PyTorch
min=10
max=20
ratio=0.5
mkdir -p logs/mntd/MLP/target/baseline-top-benign
CUDA_VISIBLE_DEVICES=1 nohup python -u mntd_train_basic_trojaned.py   \
                --task apg                     \
                --troj_type Top-benign-target  \
                --clf MLP                      \
                --min_size $min                \
                --max_size $max                \
                --max_poison_ratio $ratio      \
                > logs/mntd/MLP/target/baseline-top-benign/pytorch-backdoor-size-$min-$max-poison-0.05-$ratio-$(date "+%m.%d-%H.%M.%S").log &


#################    subset target backdoor using PyTorch, feature space      #################
families=("=mobisec" "leadbolt" "tencentprotect")
devices=(0 1 2)

for i in ${!families[@]}; do
    mkdir -p logs/mntd/MLP/target/${families[i]}/$(date "+%m%d")/
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_train_target_jp.py   \
                --task apg                     \
                --troj_type Subset             \
                --clf MLP                      \
                --subset-family ${families[i]} \
                > logs/mntd/MLP/target/${families[i]}/$(date "+%m%d")/pytorch-backdoor-poison0.001-0.002-$(date "+%m.%d-%H.%M.%S").log &
done


#################    subset target backdoor using PyTorch, problem space      #################
families=(leadbolt mobisec tencentprotect)
devices=(0 1 2)
for i in ${!families[@]}; do
    mkdir -p logs/mntd/MLP/target/${families[i]}-problem-space/$(date "+%m%d")/
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_train_target_jp.py   \
                --task apg                        \
                --troj_type Subset-problem-space  \
                --clf MLP                         \
                --subset-family ${families[i]}    \
                > logs/mntd/MLP/target/${families[i]}-problem-space/$(date "+%m%d")/pytorch-backdoor-poison0.001-0.002-$(date "+%m.%d-%H.%M.%S").log &
done



#############################################################################################
#############################################################################################
#############################################################################################
############## step 5 (Subset): train and evaluate a meta classifier, with query tuning #####

######## feature space

####### WARNING: need to train meta classifier for 1 family first, then --load_exist for other families

family=("leadbolt")
sizes=(100)
devices=(0)
# family=("mobisec"  "tencentprotect")
# sizes=(100 100)
# devices=(0 1)
type=pytorch
mkdir -p logs/mntd/MLP/meta_train/subset/$(date "+%m%d")/
setting=queryset-100-most-0-few-1-target-y-all-0-pattern-all-1
for i in ${!sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --train-num 2048    \
                    --task apg          \
                    --clf MLP           \
                    --troj_type Subset  \
                    --jumbo_max_troj_size ${sizes[i]} \
                    --method optimization-subset-expand-type0 \
                    --benign-target-model-type $type \
                    --backdoor-target-model-type $type \
                    --load_exist       \
                    --subset-family ${family[i]} \
                    --half-training 1  \
                    --debug-mode $setting \
                    > logs/mntd/MLP/meta_train/subset/$(date "+%m%d")/qt-${family[i]}-inject-p-0.001-0.002-$setting-$(date "+%m.%d-%H.%M.%S").log &
done


#############################################################################################
#############################################################################################
#############################################################################################
############## step 6 (Subset): train and evaluate a meta classifier, without query tuning ##
############## feature space

####### WANING: need to train meta for 1 family first, then --load_exist for other families
family=("leadbolt")
sizes=(100)
devices=(1)
# family=("mobisec"  "tencentprotect")
# sizes=(100 100)
# devices=(0 1)
type=pytorch
mkdir -p logs/mntd/MLP/meta_train/subset/$(date "+%m%d")/
setting=queryset-100-most-0-few-1-target-y-all-0-pattern-all-1
for i in ${!sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --train-num 2048    \
                    --task apg          \
                    --clf MLP           \
                    --troj_type Subset  \
                    --jumbo_max_troj_size ${sizes[i]} \
                    --method optimization-subset-expand-type0 \
                    --benign-target-model-type $type \
                    --backdoor-target-model-type $type \
                    --load_exist       \
                    --subset-family ${family[i]} \
                    --half-training 1  \
                    --debug-mode $setting \
                    --no_qt               \
                    > logs/mntd/MLP/meta_train/subset/$(date "+%m%d")/no-qt-${family[i]}-inject-p-0.001-0.002-$setting-$(date "+%m.%d-%H.%M.%S").log &
done




#############################################################################################
#############################################################################################
#############################################################################################
############## step 7 (Subset): evaluate the pre-trained meta classifier  ###################
############## problem space

############### with query tuning

family=("leadbolt" "mobisec" "tencentprotect")
sizes=(100 100 100);
devices=(0 1 2);
type=pytorch
mkdir -p logs/mntd/MLP/meta_train/subset-problem-space/$(date "+%m%d")/
setting=queryset-100-most-0-few-1-target-y-all-0-pattern-all-1 # WARNING:  don't change this, bind to the pre-trained meta-classifier
# setting=queryset-100-strict-larger-most-0-few-1-jumbo-2171-realizable ## if using this, we need to train another meta classifier with shadow models constructs triggers randomly from 2171 realizable features.
for i in ${!sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --train-num 2048    \
                    --task apg          \
                    --clf MLP           \
                    --troj_type Subset-problem-space  \
                    --jumbo_max_troj_size ${sizes[i]} \
                    --method optimization-subset-expand-type0 \
                    --benign-target-model-type $type \
                    --backdoor-target-model-type $type \
                    --load_exist       \
                    --subset-family ${family[i]} \
                    --half-training 1  \
                    --debug-mode $setting \
                    > logs/mntd/MLP/meta_train/subset-problem-space/$(date "+%m%d")/qt-${family[i]}-inject-p-0.001-0.002-$setting-$(date "+%m.%d-%H.%M.%S").log &
done


########## no query tuning

family=("leadbolt" "mobisec" "tencentprotect")
sizes=(100 100 100);
devices=(0 1 2);
type=pytorch
mkdir -p logs/mntd/MLP/meta_train/subset-problem-space/$(date "+%m%d")/
setting=queryset-100-most-0-few-1-target-y-all-0-pattern-all-1 # WARNING: don't change this, bind to the pre-trained meta-classifier
# setting=queryset-100-strict-larger-most-0-few-1-jumbo-2171-realizable ## if using this, we need to train another meta classifier with shadow models constructs triggers randomly from 2171 realizable features.
for i in ${!sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --train-num 2048    \
                    --task apg          \
                    --clf MLP           \
                    --troj_type Subset-problem-space  \
                    --jumbo_max_troj_size ${sizes[i]} \
                    --method optimization-subset-expand-type0 \
                    --benign-target-model-type $type \
                    --backdoor-target-model-type $type \
                    --load_exist       \
                    --subset-family ${family[i]} \
                    --half-training 1  \
                    --debug-mode $setting \
                    --no_qt               \
                    > logs/mntd/MLP/meta_train/subset-problem-space/$(date "+%m%d")/no-qt-${family[i]}-inject-p-0.001-0.002-$setting-$(date "+%m.%d-%H.%M.%S").log &
done





#############################################################################################
#############################################################################################
#############################################################################################
#################### step 8 (Baseline): train and evaluate a meta classifier ##################

############# baseline meta classifier, query tuning

num=2048
mkdir -p logs/mntd/MLP/meta_train/pytorch-top-benign/num-$num/
jumbo_sizes=(100);
target_sizes=(20);
setting=queryset-100-most-0-few-1
devices=(0);
for i in ${!jumbo_sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --load_exist       \
                    --task apg         \
                    --clf MLP           \
                    --troj_type Top-benign  \
                    --jumbo_max_troj_size ${jumbo_sizes[i]}     \
                    --target_max_troj_size ${target_sizes[i]}   \
                    --method baseline-pytorch       \
                    --benign-target-model-type pytorch   \
                    --backdoor-target-model-type pytorch   \
                    --half-training 1  \
                    --debug-mode $setting \
                    --train-num $num      \
                    > logs/mntd/MLP/meta_train/pytorch-top-benign/num-$num/qt-meta-jumbo-5-${jumbo_sizes[i]}-target-10-${target_sizes[i]}-$setting-$(date "+%m.%d-%H.%M.%S").log &
done


############ baseline meta classifier, no query tuning
# mkdir -p logs/mntd/debug/baseline/pytorch-top-benign/
num=128
mkdir -p logs/mntd/MLP/meta_train/pytorch-top-benign/num-$num/
jumbo_sizes=(100);
target_sizes=(20);
devices=(0);
setting=queryset-100-most-0-few-1
for i in ${!jumbo_sizes[@]}; do
    CUDA_VISIBLE_DEVICES=${devices[i]} nohup python -u mntd_run_meta.py   \
                    --task apg         \
                    --clf MLP           \
                    --troj_type Top-benign  \
                    --jumbo_max_troj_size ${jumbo_sizes[i]}     \
                    --target_max_troj_size ${target_sizes[i]}   \
                    --method baseline-pytorch       \
                    --benign-target-model-type pytorch   \
                    --backdoor-target-model-type pytorch   \
                    --load_exist       \
                    --half-training 1  \
                    --no_qt            \
                    --debug-mode $setting \
                    --train-num $num      \
                    > logs/mntd/MLP/meta_train/pytorch-top-benign/num-$num/no-qt-meta-jumbo-5-${jumbo_sizes[i]}-target-10-${target_sizes[i]}-$setting-$(date "+%m.%d-%H.%M.%S").log &
done
