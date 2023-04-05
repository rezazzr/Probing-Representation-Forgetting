# Probing Forgetting in Supervised and Unsupervised Continual Learning

## Poetry

[pypoetry doc](https://python-poetry.org/) is very well written and detailed.

*First, be sure to not be in a virtual env.*

To [install poetry](https://python-poetry.org/docs/#installation) with the right version :
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_VERSION=1.1.6 python`
(on Windows, from PowerShell) `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | POETRY_VERSION=1.1.6 python -
`

By default, poetry will create a virtualenv in a [cache-dir folder](https://python-poetry.org/docs/configuration/#cache-dir-string). To have it created in the repository, under a `.venv` folder, you need to first run `poetry config virtualenvs.in-project true` (https://python-poetry.org/docs/configuration/#virtualenvsin-project-boolean).
Then go to our repository, and run `poetry install`. It will create a virtualenv that can be used in PyCharm, with all the dependencies needed.

## Data
Before running the code, you need to make sure the datasets are placed in a `root` data folder, the code does not download any datasets and instead will throw an error if it cannot find the datasets.

* For **ImageNet** we need to have the following files inside the `root` data folder:
    1. `ILSVRC2012_img_val.tar`
    2. `ILSVRC2012_img_train.tar`
    3. `ILSVRC2012_img_test_v10102019.tar`
    4. `ILSVRC2012_devkit_t12.tar.gz`
* For **CUB** we need to have a folder called `CUB_200_2011` inside the `root` data folder containing the dataset.
* For **Scenes** we need to have a folder called `Scenes` inside the `root` data folder containing the dataset.


## ImageNet Transfer Experiments
In order to run these experiments, you need to follow a two-step procedure:
 1. Train a model with a CL `strategy` (either `FineTuning`, `LwF`, or `EWC`) on a sequence of tasks (`scenario`). The `scenario` has the form of `task_one2task_two`. So for the task sequence of ImageNet ➡ Scenes ➡ CUB, we would have the `scenario` of `ImageNet2Scenes2CUB`.
 2. Step 1 will save snapshots of the model at the end of each task, we can then run a linear prob model on these snapshots.

For example, running and evaluating `EWC` on `ImageNet2Scenes2CUB` would look like:
```shell
poetry run python main.py --scenario "ImageNet2Scenes2CUB" --data_root "path_to_root_data_dir" --strategy "EWC"
``` 
Now to run the linear prob evaluation at the end of training for `Scenes` and `CUB` we need to run the followings:
```shell
poetry run python main.py --probe --probe_caller "Scenes" --scenario "ImageNet2Scenes2CUB" --data_root "path_to_root_data_dir" --model_path "../../model_zoo/EWC_ImageNet2Scenes2CUB_VGG16_scenes.pt"
poetry run python main.py --probe --probe_caller "CUB" --scenario "ImageNet2Scenes2CUB" --data_root "path_to_root_data_dir" --model_path "../../model_zoo/EWC_ImageNet2Scenes2CUB_VGG16_cub.pt"
```

The `tensorboard` results will be saved in the `tb_logs` directory.