import argparse

from torchvision.datasets import ImageNet

from models.imagenet_based_models import MiniResNetSupConPretrained
from task_data_loader.imagenet import mini_train_transform, mini_valid_transform
from utilities.configs import OneShotConfig
from utilities.evaluation import PredictionBasedEvaluator
from utilities.metrics import Accuracy, Loss
from utilities.trainer import OneShotTrainer
from utilities.utils import gpu_information_summary, set_seed, xavier_uniform_initialize, safely_load_state_dict


def main(args):
    n_gpu, _ = gpu_information_summary(show=False)
    set_seed(seed_value=1609, n_gpu=n_gpu)
    print("-" * 35)
    print(f"  [Reading] ImageNet data.")
    print("-" * 35)

    # get data:
    train = ImageNet(root=args.data_root, split="train", download=None, transform=mini_train_transform)
    test = ImageNet(root=args.data_root, split="val", download=None, transform=mini_valid_transform)

    print("-" * 35)
    print(f"  [Finished] reading ImageNet data.")
    print("-" * 35)

    # Build model:
    model = MiniResNetSupConPretrained()
    if args.model_path:
        model.load_state_dict(safely_load_state_dict(args.model_path))
    else:
        model.apply(xavier_uniform_initialize)

    # train model
    trainer_config = OneShotConfig(
        prediction_evaluator=PredictionBasedEvaluator(metrics=[Accuracy(), Loss()], batch_size=512, num_workers=8),
        nb_epochs=40,
        nb_epochs_supcon=-1,
        use_sup_con=False,
        nb_classes=1000,
        batch_size=4096,
        logging_step=100,
        save_progress=True,
        progress_history=5,
        nb_warmup_steps=0,
        learning_rate=1e-3,
        experiment_name="Pretraining_MiniResNet",
        saving_dir="../../model_zoo",
        max_steps=-1,
        num_workers=8,
    )
    trainer = OneShotTrainer(model=model, config=trainer_config, train_dataset=train, valid_dataset=test)
    print("-" * 35)
    print(f"  [Starting] training.")
    print("-" * 35)

    trainer.train()

    print("-" * 35)
    print(f"  [Finished] training.")
    print("-" * 35)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="Path to the root data.",
        type=str,
    )

    parser.add_argument("--experiment_name", help="Optional name for the experiment.", type=str, default=None)
    parser.add_argument(
        "--model_path",
        help="Path of the model the CL Scenario will continue or the probing will happen on.",
        type=str,
    )
    args = parser.parse_args()

    main(args=args)
