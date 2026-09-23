import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from transformers import set_seed
from experiment.teacher_utils import launch_train_evaluate


seeds = [3470, 42, 3407, 366, 1008]

# if_train, if_evaluate, if_peft = True, True, False
# for seed in seeds:
#     set_seed(seed=seed)
#     launch_train_evaluate(seed, "bart", "HX", if_train, if_evaluate, if_peft)
#     set_seed(seed=seed)
#     launch_train_evaluate(seed, "bart", "LH", if_train, if_evaluate, if_peft)

if_train, if_evaluate, if_peft = False, True, True
for seed in seeds:
    set_seed(seed=seed)
    launch_train_evaluate(seed, "bart", "HX", if_train, if_evaluate, if_peft)
    set_seed(seed=seed)
    launch_train_evaluate(seed, "bart", "LH", if_train, if_evaluate, if_peft)
