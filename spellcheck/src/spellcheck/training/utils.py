import os
import logging

import comet_ml
from transformers.integrations.integration_utils import CometCallback


logger = logging.getLogger(__name__)


class CustomCometCallback(CometCallback):

    def setup(self, args, state, model):
        """
        Custom CometCallback to use an existing experiment during training using Trainer 
        Modification: add get_gloabl_experiment() before creating another experiment instance in CometML
        """
        self._initialized = True
        log_assets = os.getenv("COMET_LOG_ASSETS", "FALSE").upper()
        if log_assets in {"TRUE", "1"}:
            self._log_assets = True
        if state.is_world_process_zero:
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            experiment_kwargs = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            if comet_mode == "ONLINE":
                ### MODIFICATION
                experiment = comet_ml.get_global_experiment() if comet_ml.get_global_experiment() else comet_ml.Experiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml online logging enabled")
            elif comet_mode == "OFFLINE":
                experiment_kwargs["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                if hasattr(model, "config"):
                    experiment._log_parameters(model.config, prefix="config/", framework="transformers")