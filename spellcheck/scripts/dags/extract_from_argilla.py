import metaflow
from datasets import Dataset

from spellcheck.argilla.extraction import SpellcheckExtraction


class SpellcheckExtractionFromArgillaPipeline(metaflow.FlowSpec):
    """"""

    extracted_status = metaflow.Parameter(
        name="status",
        help="Which status to extract from Argilla. Can be 'submitted', 'pending', 'draft', 'discarded'",
        default="submitted",
        multiple=True # In the CLI: ... --status submitted --status pending
    )

    dataset_hf_repo = metaflow.Parameter(
        name="dataset_hf_repo",
        required=True,
        type=str,
        help="Hugging Face dataset repo id."
    )

    dataset_revision = metaflow.Parameter(
        name="dataset_revision",
        type=str,
        required=True,
        help="Uploaded dataset branch. Each branch is a revision containing the different version of the dataset."
    )

    dataset_version = metaflow.Parameter(
        name="dataset_version",
        type=str,
        required=True,
        help="New version of the dataset as a commit in the main and revision branch."
    )

    dataset_test_size = metaflow.Parameter(
        name="dataset_test_size",
        type=float,
        required=False,
        default=0,
        help="Dataset test split size used during push_to_hub. If 0, the entire dataset is labeled under 'train', meaning there is no split."
    )

    argilla_dataset_name = metaflow.Parameter(
        name="argilla_dataset_name",
        type=str,
        required=True,
        help="Dataset to extract from Argilla."
    )

    argilla_workspace_name = metaflow.Parameter(
        name="argilla_workspace_name",
        type=str,
        required=False,
        default="spellcheck",
        help="Argilla workspace name. Default to 'spellcheck'."
    )

    argilla_dataset_local_path = metaflow.Parameter(
        name="local_path",
        type=str,
        required=True,
        help="Local path to store the argilla dataset."
    )

    deploy_to_hf = metaflow.Parameter(
        name="deploy_to_hf",
        type=bool,
        required=False,
        default=False,
        help="Whether to push to dataset to HuggingFace."
    )

    additional_commit_info = metaflow.Parameter(
        name="add_info",
        type=str,
        required=False,
        default="",
        help="Whether to add a commit description to the commit."
    )

    @metaflow.step
    def start(self):
        """Start"""
        self.next(self.extract_from_argilla)

    @metaflow.step
    def extract_from_argilla(self):
        """Argilla extraction step. Takes the status as input the user wants to extract.
        """
        print("Start extraction from Argilla.")
        argilla_dataset = SpellcheckExtraction(
            dataset_name=self.argilla_dataset_name,
            workspace_name=self.argilla_workspace_name,
            extracted_status=self.extracted_status,
        ).extract_dataset()
        print(f"Extracted dataset: {argilla_dataset}")
        print(f"Save dataset in parquet format at: {self.argilla_dataset_local_path}")
        argilla_dataset.to_parquet(self.argilla_dataset_local_path)
        print("Extraction finished.")
        self.next(self.push_to_hf)

    @metaflow.step
    def push_to_hf(self):
        """Conditional step.
        
        Push the extracted dataset to a HuggingFace dataset repo.
        This step takes the version of the dataset as a commit, the revision as a branch where to push the commit.
        By default, any modification is pushed to the main branch along the revision branch.
        """
        if self.deploy_to_hf:
            print(f"Start deploying to HuggingFace. \
                        Repo_id: {self.dataset_hf_repo} - \
                            Revision: {self.dataset_revision} - \
                                Data version: {self.dataset_version}"
            )
            dataset = Dataset.from_parquet(self.argilla_dataset_local_path)
            commit_description = (
                "metaflow run id: " + metaflow.current.run_id 
                + "\n\n" + self.additional_commit_info
            )

            # Check if test_size applied
            if self.dataset_test_size > 0:
                dataset = dataset.train_test_split(test_size=self.dataset_test_size)

            # Push to main to along the revisions to show the latest push to the users
            dataset.push_to_hub(
                repo_id=self.dataset_hf_repo,
                revision="main",
                commit_message=self.dataset_version,
                commit_description=commit_description,
            )            
            # Push to revision branch
            dataset.push_to_hub(
                repo_id=self.dataset_hf_repo,
                revision=self.dataset_revision,
                commit_message=self.dataset_version,
                commit_description=commit_description,
            )
        else:
            print(f"No deployment to HF. Condition is {self.deploy_to_hf}")
            
        self.next(self.end)

    @metaflow.step
    def end(self):
        print("Pipeline finished successfully.")


if __name__ == "__main__":
    SpellcheckExtractionFromArgillaPipeline()
