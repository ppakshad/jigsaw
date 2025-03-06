######################################################################################
########################  Train SVM classifier  ##########################
mkdir -p logs/apg/SVM/
###### limit to 10000 features
iter=10000
nohup python -u backdoor_general.py -D          \
                         -R CLASSIFIER       \
                         -d apg              \
                         -c SVM              \
                         --svm-iter $iter    \
                         --n-features 10000  \
                         > logs/apg/SVM/svm-10000feature-iter$iter-$(date "+%m.%d-%H.%M.%S").log &
