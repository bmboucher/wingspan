# Minimal Terraform for running Wingspan trainers on ECS Fargate Spot.
#
# Provisions: the S3 bucket runs persist to, the IAM roles a Fargate task needs
# (an execution role to pull the image + ship logs, and a task role scoped to the
# bucket prefix for S3 read/write), an ECS cluster with the Spot capacity
# provider, and a CloudWatch log group + task definition. You build/push the
# image and launch tasks yourself (see README.md) — this is the durable infra,
# not a job scheduler.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "S3 bucket runs persist to (must be globally unique)."
  default     = "my-wingspan-runs"
}

variable "runs_prefix" {
  type        = string
  description = "Key prefix runs live under (matches s3.prefix in the run-file)."
  default     = "runs"
}

variable "image" {
  type        = string
  description = "Container image URI (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/wingspan-trainer:latest)."
}

provider "aws" {
  region = var.region
}

# ---- Persistence ----------------------------------------------------------

resource "aws_s3_bucket" "runs" {
  bucket = var.bucket_name
}

# ---- ECS cluster (with the Fargate Spot capacity provider) ----------------

resource "aws_ecs_cluster" "wingspan" {
  name = "wingspan"
}

resource "aws_ecs_cluster_capacity_providers" "wingspan" {
  cluster_name       = aws_ecs_cluster.wingspan.name
  capacity_providers = ["FARGATE_SPOT", "FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "trainer" {
  name              = "/wingspan/trainer"
  retention_in_days = 30
}

# ---- IAM: execution role (pull image + write logs) ------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "wingspan-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---- IAM: task role (the container's own S3 access) -----------------------

resource "aws_iam_role" "task" {
  name               = "wingspan-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task_s3" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.runs.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.runs.arn]
  }
}

resource "aws_iam_role_policy" "task_s3" {
  name   = "wingspan-task-s3"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_s3.json
}

# ---- Task definition ------------------------------------------------------

resource "aws_ecs_task_definition" "trainer" {
  family                   = "wingspan-trainer"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"
  memory                   = "8192"
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name        = "trainer"
      image       = var.image
      essential   = true
      stopTimeout = 120 # match the runner's graceful-stop grace + Spot's 2-min warning
      # `command` (the --config s3:// URI) is supplied per run via run-task overrides.
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.trainer.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "trainer"
        }
      }
    }
  ])
}

output "cluster" {
  value = aws_ecs_cluster.wingspan.name
}

output "task_family" {
  value = aws_ecs_task_definition.trainer.family
}

output "bucket" {
  value = aws_s3_bucket.runs.bucket
}
