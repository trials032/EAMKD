import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import statistics
from transformers import set_seed
from experiment.utils import EAMKDLoss
from experiment.student_utils import launch_train_evaluate


seeds = [3470, 42, 3407, 366, 1008]


def get_baselines_results():
    if_train, if_evaluate = True, False
    for seed in seeds:
        # set_seed(seed=seed)
        criterion, criterion_type = EAMKDLoss(8, 0.8, 0.1).cuda(), "EAMKDLoss"
        launch_train_evaluate(seed, criterion, criterion_type, "HX", if_train, if_evaluate)
        set_seed(seed=seed)
        criterion, criterion_type = EAMKDLoss(2, 0.9, 0.15).cuda(), "EAMKDLoss"
        launch_train_evaluate(seed, criterion, criterion_type, "LH", if_train, if_evaluate)


def get_unit_sensitivity_results(parameters, parameter_type):
    if_train, if_evaluate = True, True
    criterion, criterion_type = None, "EAMKDLoss"
    with open(f"./excluded_files/multi_run/sensitivity_{parameter_type}.txt", "a") as f:
        f.write(f"{parameter_type}\tHX Acc\tHX SD\tHX M-F1\tHX SD\tLH Acc\tLH SD\tLH M-F1\tLH SD\n")
    for parameter in parameters:
        HX_accs, HX_f1s, LH_accs, LH_f1s = [], [], [], []
        for seed in seeds:
            set_seed(seed=seed)
            if parameter_type == "temp":
                criterion = EAMKDLoss(parameter, 0.5, 0.5).cuda()
            if parameter_type == "lamda":
                criterion = EAMKDLoss(1, parameter, 0.5).cuda()
            if parameter_type == "gamma":
                criterion = EAMKDLoss(1, 0.5, parameter).cuda()
            acc, f1 = launch_train_evaluate(seed, criterion, criterion_type, "HX",
                                            if_train, if_evaluate, if_sensitivity=f"_{parameter_type}")
            HX_accs.append(acc)
            HX_f1s.append(f1)

            set_seed(seed=seed)
            if parameter_type == "temp":
                criterion = EAMKDLoss(parameter, 0.5, 0.5).cuda()
            if parameter_type == "lamda":
                criterion = EAMKDLoss(1, parameter, 0.5).cuda()
            if parameter_type == "gamma":
                criterion = EAMKDLoss(1, 0.5, parameter).cuda()
            acc, f1 = launch_train_evaluate(seed, criterion, criterion_type, "LH",
                                            if_train, if_evaluate, if_sensitivity=f"_{parameter_type}")
            LH_accs.append(acc)
            LH_f1s.append(f1)
        with open(f"./excluded_files/multi_run/sensitivity_{parameter_type}.txt", "a") as f:
            tmp_str = (f"{parameter}\t{round(statistics.mean(HX_accs), 2)}\t{round(statistics.stdev(HX_accs), 2)}\t"
                       f"{round(statistics.mean(HX_f1s), 2)}\t{round(statistics.stdev(HX_f1s), 2)}\t"
                       f"{round(statistics.mean(LH_accs), 2)}\t{round(statistics.stdev(LH_accs), 2)}\t"
                       f"{round(statistics.mean(LH_f1s), 2)}\t{round(statistics.stdev(LH_f1s), 2)}\n")
            f.write(tmp_str)


def get_sensitivity_results():
    temps = [0.5, 1, 2, 4, 8]
    # lamdas = [0, 0.25, 0.5, 0.75, 1]
    # gammas = [0, 0.25, 0.5, 0.75, 1]
    get_unit_sensitivity_results(temps, "temp")
    # get_unit_sensitivity_results(lamdas, "lamda")
    # get_unit_sensitivity_results(gammas, "gamma")


def get_unit_ablation_results(ablation_type):
    if_train, if_evaluate = True, True
    criterion_type = "EAMKDLoss"
    for seed in seeds:
        set_seed(seed=seed)
        criterion = EAMKDLoss(8, 0.8, 0.1, ablation_type).cuda()
        launch_train_evaluate(seed, criterion, criterion_type, "HX",
                              if_train, if_evaluate, if_ablation=f"_{ablation_type}")

        set_seed(seed=seed)
        criterion = EAMKDLoss(2, 0.9, 0.15, ablation_type).cuda()
        launch_train_evaluate(seed, criterion, criterion_type, "LH",
                              if_train, if_evaluate, if_ablation=f"_{ablation_type}")


def get_ablation_results():
    get_unit_ablation_results("CE")
    get_unit_ablation_results("CE_MSE")
    get_unit_ablation_results("CE_KL")
    get_unit_ablation_results("MSE_KL")


if "__main__" == __name__:
    get_baselines_results()
    # get_sensitivity_results()
    # get_ablation_results()
