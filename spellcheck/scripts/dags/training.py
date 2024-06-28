import os
from dotenv import load_dotenv

import metaflow
from omegaconf import OmegaConf
from sagemaker.huggingface import HuggingFace

from spellcheck.utils import get_logger, get_repo_dir
from spellcheck.argilla_modules import BenchmarkEvaluationArgilla


CONF_PATH = REPO_DIR / "config/training_llm.yml"

LOGGER = get_logger("INFO")

load_dotenv()


class TrainingPipeline(metaflow.FlowSpec):
    """Spellcheck training pipeline. 
    Model can either be trained locally, or on the cloud.
    """

    do_human_eval = metaflow.Parameter(
        "do_human_eval",
        help="Whether to push the predictions of the trained model against the benchmark to Argilla.",
        default=True,
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
    def train(self):
        """Training step.
        
        Use Sagemaker Training Job to package and run the training script in production.
        """
        # Sagemaker estimator
        estimator = HuggingFace(
            source_dir                 = self.conf.estimator.source_dir,                 # directory containing training script and requirements requirements.
            entry_point                = self.conf.estimator.entry_point,                # train script            
            dependencies               = self.conf.estimator.dependencies,               # Additional local library
            output_path                = self.conf.estimator.output_path,                # s3 path to save the artifacts
            code_location              = self.conf.estimator.code_location,              # s3 path to stage the code during the training job
            instance_type              = self.conf.estimator.instance_type,              # instances type used for the training job
            instance_count             = self.conf.estimator.instance_count,             # the number of instances used for training
            base_job_name              = self.conf.estimator.base_job_name,              # the name of the training job
            role                       = os.getenv("SAGEMAKER_ROLE"),                    # Iam role used in training job to access AWS ressources, e.g. S3
            transformers_version       = self.conf.estimator.transformers_version,       # the transformers version used in the training job
            pytorch_version            = self.conf.estimator.pytorch_version,            # the pytorch_version version used in the training job
            py_version                 = self.conf.estimator.py_version,                 # the python version used in the training job
            disable_output_compression = self.conf.estimator.disable_output_compression, # not compress output to save training time and cost
            volume_size                = self.conf.estimator.volume_size,                # the size of the EBS volume in GB           
            hyperparameters            = self.conf.hyperparameters,                      # the hyperparameters used for the training job
            environment          = {                                                     # environment variables used during training 
                "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME"),      
                "COMET_API_KEY": os.getenv("COMET_API_KEY"),      
                "EXPERIMENT_TAGS": ",".join(self.conf.estimator.comet_ml_tags),          # experiment tags list sent as a string to be JSON serialized
                "S3_MODEL_URI": self.conf.estimator.output_path,                         # the uri where the model artifact is stored is actually not in the SM_TRAINING_JOB environment variables. Let's add it.
                "S3_EVALUATION_URI": self.conf.estimator.s3_evaluation_uri,           
                "METAFLOW_RUN_ID": metaflow.current.run_id,                              # add metaflow run_id to experiment tracking tags
                "HF_TOKEN": os.getenv("HF_TOKEN"),                                       # required by some models, such as llama-3 or Mistral
            },                                                                
        )
        estimator.fit(wait=True) # Wait for the pipeline. No need for inputs since data doesn't come from S3.
        
        # Log Sagemaker training information into metaflow
        self.sagemaker_training_job_id = estimator.latest_training_job.job_name
        self.model_artifact_uri = self.conf.estimator.output_path + self.sagemaker_training_job_id
        self.evaluation_uri = os.path.join(
            self.conf.estimator.s3_evaluation_uri,
            "evaluation-" + self.sagemaker_training_job_id
        )
        self.next(self.human_evaluation)

    @metaflow.step
    def human_evaluation(self):
        """Conditional step:
        Push model predictions against the Benchmark to Argilla. 
        Evaluation dataset was generated during the training step on the Sagemaker instance.
        """
        if self.do_human_eval:
            from comet_ml import API
            experiments = API().get_experiments(
                workspace=os.getenv("COMET_WORKSPACE_NAME"),
                project_name=os.getenv("COMET_PROJECT_NAME"),
            )
            # Get the latest experiment
            self.experiment_key = experiments[-1].key      
            self.argilla_dataset_name = self.conf.estimator.base_job_name + "-exp-key-" + self.experiment_key
            BenchmarkEvaluationArgilla.from_s3(self.evaluation_uri).deploy(
                dataset_name=self.argilla_dataset_name
            )
        self.next(self.end)

    @metaflow.step
    def end(self):
        """End."""
        LOGGER.info("Training pipeline succesfully finished.")


if __name__ == "__main__":
    TrainingPipeline()
