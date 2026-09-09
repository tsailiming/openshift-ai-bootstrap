#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["eval-hub-sdk[cli]"]
# ///

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from evalhub import ModelConfig, SyncEvalHubClient
from evalhub.models.api import BenchmarkConfig, JobSubmissionRequest


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client():
    base_url = os.environ["EVALHUB_BASE_URL"]
    token = os.environ["EVALHUB_TOKEN"]
    tenant = os.environ["EVALHUB_TENANT"]

    return SyncEvalHubClient(
        base_url=base_url,
        auth_token=token,
        tenant=tenant,
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_params(values):
    params = {}

    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid parameter '{value}'. Expected KEY=VALUE."
            )

        key, raw_value = value.split("=", 1)

        try:
            params[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            params[key] = raw_value

    return params


def format_datetime(value):
    if value is None:
        return ""

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def get_job_state(job):
    state = getattr(job, "state", None)

    if state is None:
        return "unknown"

    return str(state).lower()


def get_job_id(job):
    """
    Try the common ID fields used by the SDK.
    """
    for attr in ("id", "job_id"):
        value = getattr(job, attr, None)
        if value is not None:
            return value

    resource = getattr(job, "resource", None)
    if resource is not None:
        value = getattr(resource, "id", None)
        if value is not None:
            return value

    return None


def is_terminal_state(state):
    return state in {
        "succeeded",
        "success",
        "completed",
        "complete",
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
    }


def is_success_state(state):
    return state in {
        "succeeded",
        "success",
        "completed",
        "complete",
    }


def is_failure_state(state):
    return state in {
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
    }


# ---------------------------------------------------------------------------
# Kubernetes model authentication
# ---------------------------------------------------------------------------

def create_model_secret(secret_name, api_key):
    print(f"Creating/updating Kubernetes secret: {secret_name}")

    command = [
        "oc",
        "create",
        "secret",
        "generic",
        secret_name,
        f"--from-literal=api-key={api_key}",
        "--dry-run=client",
        "-o",
        "yaml",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["oc", "apply", "-f", "-"],
            input=result.stdout,
            check=True,
            text=True,
        )

    except FileNotFoundError:
        raise RuntimeError(
            "The 'oc' command was not found. "
            "Install/configure the OpenShift CLI first."
        )


def build_model(args):
    auth = None

    if args.model_secret:
        auth = {
            "type": "kubernetes_secret",
            "name": args.model_secret,
        }

    elif args.model_api_key:
        secret_name = args.model_secret_name or f"{args.model_name}-api-key"

        create_model_secret(
            secret_name,
            args.model_api_key,
        )

        auth = {
            "type": "kubernetes_secret",
            "name": secret_name,
        }

    return ModelConfig(
        url=args.model_url,
        name=args.model_name,
        auth=auth,
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def show_providers(client):
    providers = client.providers.list()

    print(f"Providers: {len(providers)}")

    for provider in providers:
        provider_id = getattr(provider, "id", None)

        if provider_id is None:
            resource = getattr(provider, "resource", None)
            provider_id = getattr(resource, "id", None)

        print(f"\n{provider_id}")
        print(f"  Name: {getattr(provider, 'name', '')}")

        description = getattr(provider, "description", None)
        if description:
            print(f"  Description: {description}")

        benchmarks = getattr(provider, "benchmarks", None)
        if benchmarks:
            print(f"  Benchmarks: {len(benchmarks)}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def show_benchmarks(client, provider=None, category=None):
    benchmarks = client.benchmarks.list(
        provider_id=provider,
        category=category,
    )

    print(f"Benchmarks: {len(benchmarks)}")

    for benchmark in benchmarks:
        print(f"\n{benchmark.id}")
        print(f"  Name: {benchmark.name}")

        if benchmark.category:
            print(f"  Category: {benchmark.category}")

        if benchmark.description:
            print(f"  Description: {benchmark.description}")

        if benchmark.metrics:
            print(f"  Metrics: {', '.join(benchmark.metrics)}")

        print(f"  Few-shot: {benchmark.num_few_shot}")

        if benchmark.dataset_size is not None:
            print(f"  Dataset size: {benchmark.dataset_size}")

        if benchmark.tags:
            print(f"  Tags: {', '.join(benchmark.tags)}")

        if benchmark.primary_score:
            direction = (
                "lower"
                if benchmark.primary_score.lower_is_better
                else "higher"
            )

            print(
                f"  Primary score: {benchmark.primary_score.metric}"
                f" ({direction} is better)"
            )

        if benchmark.pass_criteria:
            print(
                f"  Pass threshold: "
                f"{benchmark.pass_criteria.threshold}"
            )


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

def show_collections(client):
    collections = client.collections.list()

    print(f"Collections: {len(collections)}")

    for collection in collections:
        resource = getattr(collection, "resource", None)
        collection_id = getattr(resource, "id", None)

        print(f"\n{collection_id}")
        print(f"  Name: {collection.name}")

        if collection.description:
            print(f"  Description: {collection.description}")

        category = getattr(collection, "category", None)
        if category:
            print(f"  Category: {category}")

        tags = getattr(collection, "tags", None)
        if tags:
            print(f"  Tags: {', '.join(tags)}")

        print("  Benchmarks:")

        for benchmark in collection.benchmarks:
            print(
                f"    - {benchmark.id}"
                f" (provider={benchmark.provider_id}, "
                f"weight={benchmark.weight})"
            )

            if benchmark.parameters:
                print(f"      Parameters: {benchmark.parameters}")

        pass_criteria = getattr(collection, "pass_criteria", None)
        if pass_criteria:
            print(
                f"  Pass threshold: "
                f"{pass_criteria.threshold}"
            )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

def print_job_status(job):
    job_id = get_job_id(job)
    state = get_job_state(job)

    print(f"Job: {job_id}")
    print(f"  State: {state}")

    name = getattr(job, "name", None)
    if name:
        print(f"  Name: {name}")

    created_at = getattr(job, "created_at", None)
    if created_at:
        print(f"  Created: {format_datetime(created_at)}")

    updated_at = getattr(job, "updated_at", None)
    if updated_at:
        print(f"  Updated: {format_datetime(updated_at)}")


def show_job_status(client, job_id, wait=False, interval=5):
    last_state = None

    while True:
        job = client.jobs.get(job_id)
        state = get_job_state(job)

        if state != last_state:
            if last_state is not None:
                print()

            print_job_status(job)
            last_state = state

        if not wait or is_terminal_state(state):
            break

        time.sleep(interval)

    return job


def show_all_job_status(client):
    jobs = client.jobs.list()

    print(f"Evaluation jobs: {len(jobs)}")

    if not jobs:
        return

    print()

    for job in jobs:
        job_id = get_job_id(job)
        state = get_job_state(job)
        name = getattr(job, "name", None)

        line = f"{job_id}: {state}"

        if name:
            line += f" ({name})"

        print(line)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_benchmark(client, args, model, params):
    benchmark = BenchmarkConfig(
        id=args.benchmark,
        provider_id=args.provider,
        parameters=params,
    )

    request = JobSubmissionRequest(
        name=args.name,
        model=model,
        benchmarks=[benchmark],
    )

    return client.jobs.submit(request)


def submit_collection(client, args, model, params):
    collection = client.collections.get(args.collection)

    benchmarks = []

    for benchmark in collection.benchmarks:
        benchmark_params = dict(benchmark.parameters or {})
        benchmark_params.update(params)

        benchmark_config = BenchmarkConfig(
            id=benchmark.id,
            provider_id=benchmark.provider_id,
            parameters=benchmark_params,
        )

        benchmarks.append(benchmark_config)

    request = JobSubmissionRequest(
        name=args.name,
        model=model,
        benchmarks=benchmarks,
    )

    return client.jobs.submit(request)


def submit(client, args):
    if bool(args.benchmark) == bool(args.collection):
        raise ValueError(
            "Specify exactly one of --benchmark or --collection."
        )

    params = parse_params(args.param)
    model = build_model(args)

    if args.benchmark:
        job = submit_benchmark(
            client,
            args,
            model,
            params,
        )
    else:
        job = submit_collection(
            client,
            args,
            model,
            params,
        )

    job_id = get_job_id(job)

    print(f"Submitted evaluation job: {job_id}")

    if args.wait:
        print()
        show_job_status(
            client,
            job_id,
            wait=True,
            interval=args.interval,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="CLI wrapper for EvalHub",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # providers
    # ------------------------------------------------------------------

    providers_parser = subparsers.add_parser(
        "providers",
        help="List evaluation providers",
    )

    # ------------------------------------------------------------------
    # benchmarks
    # ------------------------------------------------------------------

    benchmarks_parser = subparsers.add_parser(
        "benchmarks",
        help="List evaluation benchmarks",
    )

    benchmarks_parser.add_argument(
        "--provider",
        help="Filter benchmarks by provider ID",
    )

    benchmarks_parser.add_argument(
        "--category",
        help="Filter benchmarks by category",
    )

    # ------------------------------------------------------------------
    # collections
    # ------------------------------------------------------------------

    subparsers.add_parser(
        "collections",
        help="List evaluation collections",
    )

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    status_parser = subparsers.add_parser(
        "status",
        help="Show evaluation job status",
    )

    status_parser.add_argument(
        "job_id",
        nargs="?",
        help="Evaluation job ID",
    )

    status_parser.add_argument(
        "--all",
        action="store_true",
        dest="all_jobs",
        help="Show status for all evaluation jobs",
    )

    status_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the job reaches a terminal state",
    )

    status_parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Polling interval in seconds (default: 5)",
    )

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit an evaluation job",
    )

    target = submit_parser.add_mutually_exclusive_group(required=True)

    target.add_argument(
        "--benchmark",
        help="Benchmark ID to evaluate",
    )

    target.add_argument(
        "--collection",
        help="Collection ID to evaluate",
    )

    submit_parser.add_argument(
        "--provider",
        help="Provider ID (required for --benchmark)",
    )

    submit_parser.add_argument(
        "--name",
        required=True,
        help="Evaluation job name",
    )

    submit_parser.add_argument(
        "--model-name",
        required=True,
        help="Model name",
    )

    submit_parser.add_argument(
        "--model-url",
        required=True,
        help="Model endpoint URL",
    )

    submit_parser.add_argument(
        "--model-secret",
        help="Existing Kubernetes secret containing the model API key",
    )

    submit_parser.add_argument(
        "--model-api-key",
        help="Model API key; creates/updates a Kubernetes secret",
    )

    submit_parser.add_argument(
        "--model-secret-name",
        help="Kubernetes secret name when using --model-api-key",
    )

    submit_parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Benchmark parameter in KEY=VALUE format; can be repeated",
    )

    submit_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the submitted job reaches a terminal state",
    )

    submit_parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Polling interval in seconds (default: 5)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = get_client()

        with client:
            if args.command == "providers":
                show_providers(client)

            elif args.command == "benchmarks":
                show_benchmarks(
                    client,
                    provider=args.provider,
                    category=args.category,
                )

            elif args.command == "collections":
                show_collections(client)

            elif args.command == "status":
                if args.all_jobs and args.job_id:
                    parser.error(
                        "status: specify either JOB_ID or --all, not both"
                    )

                if not args.all_jobs and not args.job_id:
                    parser.error(
                        "status: specify JOB_ID or --all"
                    )

                if args.all_jobs:
                    if args.wait:
                        parser.error(
                            "status --all does not support --wait"
                        )

                    show_all_job_status(client)

                else:
                    show_job_status(
                        client,
                        args.job_id,
                        wait=args.wait,
                        interval=args.interval,
                    )

            elif args.command == "submit":
                if args.benchmark and not args.provider:
                    parser.error(
                        "submit --benchmark requires --provider"
                    )

                submit(client, args)

    except KeyError as exc:
        print(
            f"ERROR: Missing environment variable: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
