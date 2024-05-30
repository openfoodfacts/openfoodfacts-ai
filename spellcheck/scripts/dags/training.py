import os
from dotenv import load_dotenv

import comet_ml
from metaflow import FlowSpec, step
from omegaconf import OmegaConf
from sagemaker.huggingface import HuggingFace

# from spellcheck.jobs.train import SagemakerTrainingJob
# from spellcheck.core.schema import SagemakerEstimatorSchema, TrainerHyperparameters
from spellcheck.utils import get_logger, get_repo_dir


REPO_DIR = get_repo_dir()
CONF_PATH = REPO_DIR / "config/training.yml"

LOGGER = get_logger()

load_dotenv()


class TrainingPipeline(FlowSpec):
    """"""
    @step
    def start(self):
        self.conf = OmegaConf.load(CONF_PATH)
        LOGGER.info(f"Config: {self.conf}")
        self.next(self.train)

    @step
    def train(self):
        """"""
        estimator = HuggingFace(
            source_dir           = self.conf.estimator.source_dir,            # directory containing training script and requirements requirements.
            entry_point          = self.conf.estimator.entry_point,           # train script            
            dependencies         = self.conf.estimator.dependencies,          # Additional local library
            output_path          = self.conf.estimator.output_path,           # s3 path to save the artifacts
            code_location        = self.conf.estimator.code_location,         # s3 path to stage the code during the training job
            instance_type        = self.conf.estimator.instance_type,         # instances type used for the training job
            instance_count       = self.conf.estimator.instance_count,        # the number of instances used for training
            base_job_name        = self.conf.estimator.base_job_name,         # the name of the training job
            role                 = os.getenv("SAGEMAKER_ROLE"),                  # Iam role used in training job to access AWS ressources, e.g. S3
            transformers_version = self.conf.estimator.transformers_version,  # the transformers version used in the training job
            pytorch_version      = self.conf.estimator.pytorch_version,       # the pytorch_version version used in the training job
            py_version           = self.conf.estimator.py_version,            # the python version used in the training job
            environment          = {
                "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME"),
                "COMET_API_KEY": os.getenv("COMET_API_KEY"),
                "EXPERIMENT_TAGS": self.conf.estimator.comet_ml_tags,
            },                                                                 # environment variables used during training 
            hyperparameters      = self.conf.hyperparameters,                 # the hyperparameters used for the training job
        )
        estimator.fit(wait=True) # Wait for the pipeline
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":

    TrainingPipeline()