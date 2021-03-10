import os
import pytest
import subprocess
from ast import literal_eval
from sagemaker.huggingface import HuggingFace
from sagemaker import TrainingJobAnalytics

# from sagemaker.huggingface import HuggingFace

# TODO add in setup.py there is a release
# | bert-finetuning-pytorch       | testbertfinetuningusingBERTfromtransformerlib+PT                  | SageMaker createTrainingJob | 1        | Accuracy, time to train |
# |-------------------------------|-------------------------------------------------------------------|-----------------------------|----------|-------------------------|
# | bert-finetuning-pytorch-ddp   | test bert finetuning using BERT from transformer lib+ PT DPP      | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | bert-finetuning-pytorch-smddp | test bert finetuning using BERT from transformer lib+ PT SM DDP   | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | deberta-finetuning-smmp       | test deberta finetuning using BERT from transformer lib+ PT SM MP | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # current DLCs are only in us-east-1 available
os.environ["AWS_PROFILE"] = "hf-sm"  # local profile FIXME: needs to be removed to work in the pipeline

SAGEMAKER_ROLE = "arn:aws:iam::558105141721:role/sagemaker_execution_role"
ECR_IMAGE = "564829616587.dkr.ecr.us-east-1.amazonaws.com/huggingface-training:pytorch1.6.0-transformers4.3.1-tokenizers0.10.1-datasets1.2.1-py36-gpu-cu110"


@pytest.mark.skipif(
    literal_eval(os.getenv("TEST_SAGEMAKER", "False")) is not True,
    reason="Skipping test because should only be run when releasing minor transformers version",
)
@pytest.mark.parametrize("model_name_or_path", ["distilbert-base-cased"])
@pytest.mark.parametrize("instance_count", [1])
@pytest.mark.parametrize("instance_type", ["ml.g4dn.xlarge"])
def test_single_node_fine_tuning(instance_type, instance_count, model_name_or_path):
    print("instance_type", instance_type)
    print("instance_count", instance_count)
    print("model_name_or_path", model_name_or_path)
    # cannot use git since, we need the requirements.txt to install the newest transformers version
    subprocess.run(
        "cp ./examples/text-classification/run_glue.py ./tests/sagemaker/scripts/run_glue.py".split(),
        encoding="utf-8",
        check=True,
    )
    # defines hyperparameters
    hyperparameters = {
        "model_name_or_path": model_name_or_path,
        "task_name": "mnli",
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "num_train_epochs": 2,
        "output_dir": "/opt/ml/model",
        "overwrite_output_dir": True,
        "max_steps": 500,
        "save_steps": 5500,
    }
    # metric definition to extract the results
    metric_definitions = [
        {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
        {"Name": "eval_accuracy", "Regex": "eval_accuracy.*=\D*(.*?)$"},
        {"Name": "eval_loss", "Regex": "eval_loss.*=\D*(.*?)$"},
    ]
    # creates estimator
    estimator = HuggingFace(
        entry_point="run_glue_old.py",
        source_dir="./tests/sagemaker/scripts",
        role=SAGEMAKER_ROLE,
        image_uri=ECR_IMAGE,
        instance_count=instance_count,
        instance_type=instance_type,
        debugger_hook_config=False,
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
    )
    # run training
    estimator.fit()

    # test csv
    TrainingJobAnalytics(estimator.latest_training_job.name).export_csv("metrics.csv")

    result_metrics_df = TrainingJobAnalytics(estimator.latest_training_job.name).dataframe()

    train_runtime = list(result_metrics_df[result_metrics_df.metric_name == "train_runtime"]["value"])
    eval_accuracy = list(result_metrics_df[result_metrics_df.metric_name == "eval_accuracy"]["value"])
    eval_loss = list(result_metrics_df[result_metrics_df.metric_name == "eval_loss"]["value"])

    assert all(t <= 200 for t in train_runtime)
    assert all(t >= 0.6 for t in eval_accuracy)
    assert all(t <= 0.9 for t in eval_loss)


# @pytest.mark.skipif(
#     literal_eval(os.getenv("TEST_SAGEMAKER", "False")) is not True,
#     reason="Skipping test because should only be run when releasing minor transformers version",
# )
# @pytest.mark.parametrize("model_name_or_path", ["distilbert-base-uncased"])
# @pytest.mark.parametrize("instance_types", ["ml.p3dn.24xlarge"])
# @pytest.mark.parametrize("instance_count", [2, 4, 8, 16])
# def test_multi_node_ddp_fine_tuning(instance_types, instance_count, model_name_or_path):
#     """
#     Tests smddprun command via Estimator API distribution parameter
#     """
#     pass
#     distribution = {"smdistributed":{"dataparallel":{"enabled":True}}}
#     estimator = PyTorch(entry_point='smdataparallel_mnist.py',
#                         role='SageMakerRole',
#                         image_uri=ecr_image,
#                         source_dir=mnist_path,
#                         instance_count=2,
#                         instance_type=instance_types,
#                         sagemaker_session=sagemaker_session,
#                         debugger_hook_config=False,
#                         distribution=distribution)

#     estimator.fit()