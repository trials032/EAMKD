import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import re
import shutil
from transformers import set_seed
from create_dataset.utils import get_distilled_dataset


seeds = [3470, 42, 3407, 366, 1008]


def get_checkpoint(dataset_name):
    pattern = r"Random Seed\s*=\s*\d+\s*\nCheckpoint\s+(\d+)"
    teacher_names = ["roberta", "deberta", "bart"]
    checkpoints = []
    final_checkpoints = []
    for teacher_name in teacher_names:
        with open(f"../excluded_files/multi_run/teacher_{teacher_name}_{dataset_name}.txt", "r") as f:
            content = f.read()
        matches = re.findall(pattern, content, re.DOTALL)
        checkpoints.append([int(x) for x in matches])
    for a, b, c in zip(checkpoints[0], checkpoints[1], checkpoints[2]):
        tmp_dict = {teacher_names[0]: a, teacher_names[1]: b, teacher_names[2]: c}
        print(tmp_dict)
        final_checkpoints.append(tmp_dict)
    return final_checkpoints


HX_checkpoints = get_checkpoint("HX")
for seed, checkpoint in zip(seeds, HX_checkpoints):
    folder_path = f"../excluded_files/dataset/HX_distilled_{seed}"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    set_seed(seed=seed)
    dataset_distilled = get_distilled_dataset("HX", checkpoint, seed)
    dataset_distilled.save_to_disk(folder_path)

LH_checkpoints = get_checkpoint("LH")
for seed, checkpoint in zip(seeds, LH_checkpoints):
    folder_path = f"../excluded_files/dataset/LH_distilled_{seed}"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    set_seed(seed=seed)
    dataset_distilled = get_distilled_dataset("LH", checkpoint, seed)
    dataset_distilled.save_to_disk(folder_path)
