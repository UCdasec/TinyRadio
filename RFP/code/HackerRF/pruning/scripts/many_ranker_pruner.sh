python ranker_pruner.py prune-automatic ~/TinyRadio/RFP/experiments/ResNet/HackerRF/BASELINE/best_checkpoint.h5 RYAN_RANKER_PRUNER_RESULTS/ranker_pruner_n2 l2 2 --show-model-summary --verbose --no-training-verbose
python ranker_pruner.py prune-automatic ~/TinyRadio/RFP/experiments/ResNet/HackerRF/BASELINE/best_checkpoint.h5 RYAN_RANKER_PRUNER_RESULTS/ranker_pruner_n4 l2 4 --show-model-summary --verbose --no-training-verbose
python ranker_pruner.py prune-automatic ~/TinyRadio/RFP/experiments/ResNet/HackerRF/BASELINE/best_checkpoint.h5 RYAN_RANKER_PRUNER_RESULTS/ranker_pruner_n8 l2 8 --show-model-summary --verbose --no-training-verbose

