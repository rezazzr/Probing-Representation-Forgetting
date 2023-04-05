from pprint import pprint
import argparse
import task_data_loader.imagenet
from models.imagenet_based_models import PredictionLayerConfig
from task_data_loader.split_cifar10 import cifar10_basic_transform
from task_data_loader.scenarios import (
    SplitCIFAR10,
    ImageNet2CUB,
    ImageNetScenario,
    ImageNet2Scenes,
    ImageNet2Scenes2CUB,
    ScenesScenario,
    ImageNet2Flowers,
    ImageNet2Scenes2CUB2Flowers,
    DuplicatedHalfCIFAR10,
)
from utilities.configs import TrainingConfig
from utilities.evaluation import RepresentationBasedEvaluator, PredictionBasedEvaluator
from utilities.metrics import CKA, L2, Accuracy, Loss
from models import cifar10, imagenet_based_models
from utilities.trainer import ModelCoach, ProbeEvaluator
from utilities.utils import gpu_information_summary, set_seed, EarlyStoppingConfig
from utilities.utils import xavier_uniform_initialize
import torch

TASK_META_DATA = {"CUB": 200, "Scenes": 67, "ImageNet": 1000, "Flowers": 102}


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=args.seed_value, n_gpu=n_gpu)

    print("-" * 35)
    print(f"  Testing the {args.scenario} scenario.")
    print("-" * 35)

    if not args.probe:
        # load the proper CL Scenario
        if args.scenario == "SplitCIFAR10":
            cl_task = SplitCIFAR10(root=args.data_root, transforms=cifar10_basic_transform)
        elif args.scenario == "DuplicatedHalfCIFAR10":
            cl_task = DuplicatedHalfCIFAR10(root=args.data_root, transforms=cifar10_basic_transform)
        elif args.scenario.startswith("ImageNet"):
            if args.backbone == "MiniResNet":
                transforms = [
                    task_data_loader.imagenet.mini_train_transform,
                    task_data_loader.imagenet.mini_valid_transform,
                ]
            else:
                transforms = [task_data_loader.imagenet.train_transform, task_data_loader.imagenet.valid_transform]

            if args.scenario == "ImageNet2CUB":
                cl_task = ImageNet2CUB(root=args.data_root, transforms=transforms)
            elif args.scenario == "ImageNet2Scenes":
                cl_task = ImageNet2Scenes(root=args.data_root, transforms=transforms)
            elif args.scenario == "ImageNet2Scenes2CUB":
                cl_task = ImageNet2Scenes2CUB(root=args.data_root, transforms=transforms)
            elif args.scenario == "ImageNet2Flowers":
                cl_task = ImageNet2Flowers(root=args.data_root, transforms=transforms)
            elif args.scenario == "ImageNet2Scenes2CUB2Flowers":
                cl_task = ImageNet2Scenes2CUB2Flowers(root=args.data_root, transforms=transforms)
        else:
            raise Exception("Scenario is not supported!", args.scenario)

        # load the proper model
        if args.scenario.startswith("ImageNet"):
            tasks_in_scenario = args.scenario.split("2")
            prediction_layer = [
                PredictionLayerConfig(task_id=task.lower(), nb_classes=TASK_META_DATA[task])
                for task in tasks_in_scenario[1:]
            ]
            if args.backbone == "VGG":
                model = imagenet_based_models.VGG16(prediction_layers=prediction_layer)
            elif args.backbone == "MiniResNet":
                if args.supcon:
                    model = imagenet_based_models.MiniResNetSupCon(
                        back_bone_path=args.model_path, prediction_layers=prediction_layer
                    )
                else:
                    model = imagenet_based_models.MiniResNet(
                        back_bone_path=args.model_path, prediction_layers=prediction_layer
                    )
            else:
                raise Exception("The backbone model is not supported for CIFAR10 data. --backbone: ", args.backbone)
        else:
            if args.backbone == "ResNet":
                model = cifar10.ResNet()
            elif args.backbone == "VGG":
                model = cifar10.VGG()
            else:
                raise Exception("The backbone model is not supported for CIFAR10 data. --backbone: ", args.backbone)

            model.apply(xavier_uniform_initialize)

        training_config = TrainingConfig(
            batch_size=256,
            num_workers=30,
            logging_step=1000,
            early_stopping_config=None,
            nb_epochs=500,
            use_scheduler=True,
            nb_warmup_steps=200,
            learning_rate=1e-5,
            max_steps=-1,
            prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy(), Loss()], batch_size=512, num_workers=8),
            representation_evaluator=None,
            is_probe=False,
            save_progress=True,
            saving_dir="../../model_zoo",
            strategy=args.strategy,
            experiment_name=args.experiment_name,
            use_different_seed=args.use_different_seed,
            seed_value=args.seed_value,
            use_sup_con=args.supcon,
            nb_epochs_supcon=100,
        )
        trainer = ModelCoach(model=model, data_stream=cl_task, config=training_config)
        training_results_last_iter = trainer.train()
        pprint(training_results_last_iter)
    else:
        tasks_in_scenario = args.scenario.split("2")
        if tasks_in_scenario[0] == "ImageNet":
            if args.probing_train_data == "ImageNet":
                if args.backbone == "MiniResNet":
                    transforms = [
                        task_data_loader.imagenet.mini_train_transform,
                        task_data_loader.imagenet.mini_valid_transform,
                    ]
                else:
                    transforms = [task_data_loader.imagenet.train_transform, task_data_loader.imagenet.valid_transform]
                cl_task = ImageNetScenario(root=args.data_root, transforms=transforms)
            elif args.probing_train_data == "Scenes":
                cl_task = ScenesScenario(root=args.data_root)
            else:
                raise Exception("The --probing_train_data is not supported.", args.probing_train_data)

            training_config = TrainingConfig(
                batch_size=4096,
                num_workers=30,
                logging_step=4000,
                early_stopping_config=None,
                nb_epochs=20,
                use_scheduler=True,
                nb_warmup_steps=200,
                learning_rate=0.003,
                max_steps=-1,
                prediction_evaluator=PredictionBasedEvaluator(
                    metrics=[Accuracy(), Loss()], batch_size=4096, num_workers=30
                ),
                is_probe=True,
                save_progress=True,
                saving_dir="probe_zoo",
                experiment_name=args.experiment_name,
                seed_value=args.seed_value,
            )

            probe_evaluator = ProbeEvaluator(
                blocks_to_prob=["block0"],
                data_stream=cl_task,
                half_precision=True,
                prediction_layers=[
                    PredictionLayerConfig(
                        task_id=args.probing_train_data.lower(), nb_classes=TASK_META_DATA[args.probing_train_data]
                    ),
                ],
                training_configs=training_config,
            )
            prediction_layer = [
                PredictionLayerConfig(task_id=task.lower(), nb_classes=TASK_META_DATA[task])
                for task in tasks_in_scenario[1:]
            ]

            if args.backbone == "VGG":
                model = imagenet_based_models.VGG16(prediction_layers=prediction_layer)
            elif args.backbone == "MiniResNet":
                if args.supcon:
                    model = imagenet_based_models.MiniResNetSupCon(
                        back_bone_path=None, prediction_layers=prediction_layer
                    )
                else:
                    model = imagenet_based_models.MiniResNet(back_bone_path=None, prediction_layers=prediction_layer)
            else:
                raise Exception("The backbone model is not supported for CIFAR10 data. --backbone: ", args.backbone)

            model.load_state_dict(torch.load(args.model_path))
            probe_results = probe_evaluator.probe(model=model, probe_caller=args.probe_caller)
            pprint(probe_results)
        else:
            raise Exception("Scenario is not supported!", args.scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        help="Probes a model rather than training a CL scenario",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model_path",
        help="Path of the model the CL Scenario will continue or the probing will happen on.",
        type=str,
    )
    parser.add_argument(
        "--scenario",
        help="Name of the CL scenario, indicative of task sequence.",
        type=str,
    )
    parser.add_argument(
        "--probe_caller",
        help="Indicates which task is calling the prob.",
        type=str,
    )

    parser.add_argument(
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument(
        "--strategy",
        help="Name of the CL strategy, indicative of forgetting mitigation strategy",
        type=str,
        choices=["FineTuning", "LwF", "EWC8000", "EWC500"],
        default="FineTuning",
    )

    parser.add_argument(
        "--probing_train_data",
        help="Indicates the training dataset for the linear prob",
        type=str,
        choices=["ImageNet", "Scenes"],
        default="ImageNet",
    )

    parser.add_argument(
        "--backbone",
        help="Either VGG or ResNet backbone.",
        type=str,
        choices=["VGG", "ResNet", "MiniResNet"],
        default="ResNet",
    )

    parser.add_argument("--experiment_name", help="Optional name for the experiment.", type=str, default=None)
    parser.add_argument(
        "--use_different_seed",
        help="Indicates if we want to use different seeds for different models.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--seed_value",
        help="Random seed value for this experiment.",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--supcon",
        help="Whether to use supcon or not.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    main(args=args)
