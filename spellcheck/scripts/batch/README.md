# Google Batch job

## Notes

* Netherland (europe-west4) has GPUs (A100, L4)
* Check [CLOUD-LOGGING](https://console.cloud.google.com/logs/query;query=SEARCH%2528%22spellcheck%22%2529;cursorTimestamp=2024-08-14T11:21:32.485988660Z;duration=PT1H?referrer=search&project=robotoff) for logs
* Require deep learning image to run: [deep learning containers list](https://cloud.google.com/deep-learning-containers/docs/choosing-container#pytorch)
* Custom storage capacity to host the heavy docker image (~24GB) by adding BootDisk
* 1000 products processed: 1:30min (g2-instance-with 8) (overall batch job: 3:25min)
    * L4: g2-instance-8 hourly cost: $0.896306 -> ~ 0.05$ to process batch of 1000
    * A100: a2-highgpu-1g: $3.748064
* A100/Cuda doesn't support FP8
* A100 has less availability than L4: need to wait for batch job (10+ min)

## Links

* [GPU availability per region](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)
* [Batch job with GPU](https://cloud.google.com/batch/docs/create-run-job-gpus#create-job-gpu-examples)
* [VM Instance pricing](https://cloud.google.com/compute/vm-instance-pricing#vm-instance-pricing)

## Commands

### List GPUs per region
```bash
gcloud compute accelerator-types list
```

### List deep learning images
```bash
gcloud compute images list \
--project deeplearning-platform-release \
--format="value(NAME)" \
--no-standard-images
```

## Workflow / Orchestration

* [Workflow](https://cloud.google.com/workflows/docs/overview)
