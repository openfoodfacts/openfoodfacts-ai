import os
from dotenv import load_dotenv

import metaflow
from omegaconf import OmegaConf
from sagemaker.huggingface import HuggingFace

# from spellcheck.jobs.train import SagemakerTrainingJob
# from spellcheck.core.schema import SagemakerEstimatorSchema, TrainerHyperparameters
from spellcheck.utils import get_logger, get_repo_dir
from spellcheck.argilla_modules import BenchmarkEvaluationArgilla


REPO_DIR = get_repo_dir()
CONF_PATH = REPO_DIR / "config/training.yml"

LOGGER = get_logger()

load_dotenv()


class TrainingPipeline(metaflow.FlowSpec):
    """Spellcheck training pipeline. 
    Model can either be trained locally, or on the cloud.
    """

    do_human_eval = metaflow.Parameter(
        "do_human_eval",
        help="Whether to push the predictions of the trained model against the benchmark to Argilla.",
        default=False,
        type=bool,
    )

    @metaflow.step
    def start(self):
        """Load all parameters from config file used during training. 
        """
        self.conf = OmegaConf.load(CONF_PATH)
        LOGGER.info(f"Config parameters: {self.conf}")
        self.next(self.train)

    @metaflow.step
    def validate(self):
        """Validation step.

        Validate: 
            - cloud instance configuration,
            - training hyperparameters,
            - datasets schemas. 
        """
        pass

    @metaflow.step
    def train(self):
        """Training step.
        
        Use Sagemaker Training Job to package and run the training script in production.
        """
        self.estimator = HuggingFace(
            source_dir           = self.conf.estimator.source_dir,            # directory containing training script and requirements requirements.
            entry_point          = self.conf.estimator.entry_point,           # train script            
            dependencies         = self.conf.estimator.dependencies,          # Additional local library
            output_path          = self.conf.estimator.output_path,           # s3 path to save the artifacts
            code_location        = self.conf.estimator.code_location,         # s3 path to stage the code during the training job
            instance_type        = self.conf.estimator.instance_type,         # instances type used for the training job
            instance_count       = self.conf.estimator.instance_count,        # the number of instances used for training
            base_job_name        = self.conf.estimator.base_job_name,         # the name of the training job
            role                 = os.getenv("SAGEMAKER_ROLE"),               # Iam role used in training job to access AWS ressources, e.g. S3
            transformers_version = self.conf.estimator.transformers_version,  # the transformers version used in the training job
            pytorch_version      = self.conf.estimator.pytorch_version,       # the pytorch_version version used in the training job
            py_version           = self.conf.estimator.py_version,            # the python version used in the training job
            hyperparameters      = self.conf.hyperparameters,                 # the hyperparameters used for the training job            
            environment          = {                                          # environment variables used during training 
                "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME"),
                "COMET_API_KEY": os.getenv("COMET_API_KEY"),
                "EXPERIMENT_TAGS": (                                          # add Metaflow run_id to the training job
                    self.conf.estimator.comet_ml_tags 
                    + metaflow.current.run_id                                 # add metaflow run_id to experiment tracking tags
                ),
                "S3_OUTPUT_URI": self.conf.estimator.output_path,             # the uri where the model artifact is stored is actually not in the SM_TRAINING_JOB environment variables. Let's add it.                                                            
            },                                                                
        )
        self.estimator.fit(wait=True) # Wait for the pipeline. No need for inputs since data doesn't come from S3.
        
        # Log estimator information into metaflow
        self.sagemaker_training_job_id = self.estimator.latest_training_job.job_name
        
        self.next(self.end)

    @metaflow.step
    def human_evaluation(self):
        """Conditional step.
        Push predictions of the trained model to Argilla.
        """
        pass


    @metaflow.step
    def end(self):
        """Cleaning."""

if __name__ == "__main__":

    TrainingPipeline()