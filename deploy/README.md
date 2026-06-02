# Cloud training (containerized, S3-persisted, Spot-friendly)

Run Wingspan self-play training unattended in a container, with all persistence
in S3, resumable after interruption, and watchable across runs. One YAML run-file
configures a run; the same file relaunched resumes it.

## How it works

`wingspan.cloud` wraps the existing `TrainingLoop` (the FLYWAY CONTROL worker)
with an S3 sync sidecar and runs it headless:

- **One run-file** (`run.example.yaml`) holds everything: `run_name`, the `s3`
  target (bucket/prefix/region — **no credentials**), the `sync` cadences, and a
  `train` block that maps 1:1 onto `TrainConfig`.
- **Persistence** lives at `s3://<bucket>/<prefix>/<run_name>/`:
  `last.pt` / `best.pt` / `opponent.pt` / `setup.pt`, `metrics.jsonl`,
  `model_config.json`, a frequently-refreshed `status.json`, the per-game log as
  immutable chunks under `games/<session>/chunk_*.jsonl`, and at the target
  milestone `final_<n>.pt` + `final_eval_<n>.json`.
- **No S3 spam:** the ~1 KB status uploads on a wall-clock interval; the
  checkpoint set every `checkpoint_upload_iters`; the high-volume game log only
  as size-bounded chunks. Per-iteration *local* writes are unchanged.
- **Interruptible:** `SIGTERM` (a Spot reclaim, ~2 min warning) or `SIGINT`
  triggers a graceful stop — finish the current game, write the final checkpoint,
  do a closing S3 sync — then exit. Relaunch with the same run-file and it pulls
  state back from S3 and continues from the stored iteration.
- **Terminates at the target:** at `target_iterations` it runs the large
  fixed-model eval, writes `final_eval_<n>.json` to S3, and exits 0.

Auth is the **IAM task role** on Fargate (or your env / `~/.aws` locally) — the
run-file never holds secrets.

## Build the image

```
docker build -t wingspan-trainer .
```

## Run locally (or for a smoke test against MinIO)

```
docker run --rm -v "$PWD/deploy/run.example.yaml:/config/run.yaml" \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  wingspan-trainer
```

To target a local MinIO, set `s3.endpoint_url` in the run-file (or pass
`--endpoint-url http://host.docker.internal:9000` after the image name).

## Run on ECS Fargate Spot

1. Provision infra (one time): edit and `terraform apply` `main.tf`
   (S3 bucket, ECS cluster + Spot capacity provider, task roles, log group, task
   definition). Push the image to ECR.
2. Upload the run-file once: `aws s3 cp deploy/run.example.yaml
   s3://<bucket>/configs/<run>.yaml`.
3. Launch a Spot task, pointing the container at that key:

```
aws ecs run-task --cluster wingspan --task-definition wingspan-trainer \
  --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-xxxx],securityGroups=[sg-xxxx],assignPublicIp=ENABLED}' \
  --overrides '{"containerOverrides":[{"name":"trainer","command":["--config","s3://<bucket>/configs/<run>.yaml"]}]}'
```

When Spot reclaims the task, just run it again with the **same command** — it
resumes from S3. The task definition sets `stopTimeout: 120` so the graceful
stop + final sync complete inside Spot's warning window.

`ecs-task-def.json` is a standalone equivalent of the Terraform task definition
for `aws ecs register-task-definition --cli-input-json file://...` if you prefer
not to use Terraform.

## Monitor all runs

```
wingspan-monitor --bucket <bucket> --prefix runs --region <region>
```

A live roster reading each run's `status.json`: an in-flight LED (fresh heartbeat
+ non-terminal phase), % of target, total games, average score, win rate vs the
current challenger, and ETA. Read-only; refreshes every 10 s (`--interval`).
