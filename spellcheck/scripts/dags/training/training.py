import os
from dotenv import load_dotenv

import metaflow

from spellcheck.utils import get_logger, get_repo_dir
from spellcheck.argilla.deployment import BenchmarkEvaluationArgilla


REPO_DIR = get_repo_dir()
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

    training_conf_path = metaflow.Parameter(
        "training_conf_path",
        default=REPO_DIR / "config/training/training_conf.yml",
        help="Path to the Sagemaker estimator configuration file.",
    )

    experiment_tags = metaflow.Parameter(
        "experiment_tag",
        required=False,
        type=str,
        multiple=True,
        help="Tags to add to the CometML experiment.",
    )

    training_data_version = metaflow.Parameter(
        "training_data_version",
        required=True,
        type=str,
        help="Version of the training dataset used during training.",
    )

    evaluation_data_version = metaflow.Parameter(
        "evaluation_data_version",
        required=True,
        type=str,
        help="Version of the evaluation dataset used during training."
    )

    @metaflow.step
    def start(self):
        """Load all parameters from config file used during training. 
        """
        import comet_ml
        # Create experiment in CometML and log information before starting the training job
        experiment = comet_ml.Experiment()
        experiment.add_tags(list(self.experiment_tags))
        experiment.log_parameters({
            "metaflow_run_id": metaflow.current.run_id,
            "training_data_version": self.training_data_version,
            "evaluation_data_version": self.evaluation_data_version,
        })
        self.experiment_key = experiment.get_key()
        experiment.end()
        self.next(self.train)

    @metaflow.step
    def train(self):
        """Training step.
        
        Use Sagemaker Training Job to package and run the training script in production.
        """
        from sagemaker.huggingface import HuggingFace
        from omegaconf import OmegaConf
        import comet_ml

        LOGGER.info(f"Configuration file used: {self.training_conf_path}")
        self.training_conf = OmegaConf.load(self.training_conf_path)
        hyperparameters = OmegaConf.to_container(self.training_conf.hyperparameters, resolve=True) # Transform DictConfig to Dict
        LOGGER.info(f"Configs: {self.training_conf}")

        # CometML
        experiment = comet_ml.ExistingExperiment(previous_experiment=self.experiment_key)
        experiment.log_parameters(hyperparameters)
        experiment.log_code(file_name=os.path.join(
            self.training_conf.estimator.source_dir, 
            self.training_conf.estimator.entry_point
        ))
        # Save config as code
        experiment.log_code(file_name=self.training_conf_path)

        # Prepare Sagemaker estimator
        estimator = HuggingFace(
            role= os.getenv("SAGEMAKER_ROLE"),                                              # Iam role used in training job to access AWS ressources, e.g. S3
            hyperparameters= hyperparameters,                                               # hyperparameters used for the training job
            environment={                                                                   # environment variables used during training 
                "COMET_PROJECT_NAME": os.getenv("COMET_PROJECT_NAME"),                      # comet project name
                "COMET_API_KEY": os.getenv("COMET_API_KEY"),   
                "COMET_EXPERIMENT_KEY": self.experiment_key,        
                "HF_TOKEN": os.getenv("HF_TOKEN"),                                          # required by some models, such as llama-3 or Mistral
                "S3_MODEL_URI": self.training_conf.estimator.output_path,                   # the uri where the model artifact is stored is actually not in the SM_TRAINING_JOB environment variables. Let's add it.
                # "S3_EVALUATION_URI": self.training_conf.additional_conf.s3_evaluation_uri,  # s3 uri where evaluation prediction are stored        
            },
            **self.training_conf.estimator,                                        
        )

        # Run training job
        estimator.fit(wait=True) # Add previous model
        
        # Log Sagemaker training information into metaflow after training job
        self.sagemaker_training_job_id = estimator.latest_training_job.job_name
        self.model_artifact_uri = self.training_conf.estimator.output_path + self.sagemaker_training_job_id
        # self.evaluation_uri = os.path.join(
        #     self.training_conf.additional_conf.s3_evaluation_uri,
        #     "evaluation-" + self.sagemaker_training_job_id
        # )

        # Log model artifact in cometml
        experiment.log_remote_model(model_name="model", model_path=self.model_artifact_uri, sync_mode=False)
        experiment.log_parameter("training_job_name", self.sagemaker_training_job_id)
        
        self.next(self.human_evaluation)

    @metaflow.step
    def human_evaluation(self):
        """Conditional step:
        Push model predictions against the Benchmark to Argilla. 
        Evaluation dataset was generated during the training step on the Sagemaker instance.
        """
        if self.do_human_eval:
            # Get the latest experiment
            self.argilla_dataset_name = self.training_conf.estimator.base_job_name + "-exp-key-" + self.experiment_key
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
