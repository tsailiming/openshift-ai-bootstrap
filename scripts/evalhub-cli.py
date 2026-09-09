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

from evalhub import JobLogOptions, ModelConfig, SyncEvalHubClient
from evalhub.cli.formatter import output
from evalhub.client.job_logs import is_terminal_job
from evalhub.models import ModelAuth
from evalhub.models.api import BenchmarkConfig, JobStatus, JobSubmissionRequest


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
    effective_state = getattr(job, "effective_state", None)

    if effective_state is not None:
        if hasattr(effective_state, "value"):
            return effective_state.value

        return str(effective_state).lower()

    state = getattr(job, "state", None)

    if state is None:
        return "unknown"

    if hasattr(state, "value"):
        return state.value

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
    secret_name = None

    if args.model_secret:
        secret_name = args.model_secret

    elif args.model_api_key:
        secret_name = args.model_secret_name or f"{args.model_name}-api-key"

        create_model_secret(
            secret_name,
            args.model_api_key,
        )

    auth = ModelAuth(secret_ref=secret_name) if secret_name else None

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

        if not wait or is_terminal_job(job):
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


def cancel_job(client, job_id, hard_delete=False):
    client.jobs.cancel(job_id, hard_delete=hard_delete)

    action = "deleted" if hard_delete else "cancelled"
    print(f"Job {job_id} {action}.")


def show_job_logs(client, args):
    options = JobLogOptions(
        tail_lines=args.tail,
        timestamps=args.timestamps,
        since_seconds=args.since,
    )

    if args.follow:
        print(f"Streaming logs for job {args.job_id}...", file=sys.stderr)

        final_state = None

        try:
            for update in client.jobs.watch_logs(
                args.job_id,
                benchmark_index=args.benchmark_index,
                options=options,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            ):
                if update.logs:
                    print(update.logs, end="")

                final_state = update.job.effective_state

        except TimeoutError:
            print(
                f"\nTimed out before job {args.job_id} reached a terminal state.",
                file=sys.stderr,
            )
            sys.exit(2)

        print(file=sys.stderr)

        if final_state not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.PARTIALLY_FAILED,
        }:
            print(
                f"Stream ended before completion (last state: {final_state}).",
                file=sys.stderr,
            )
            sys.exit(2)

        print(
            f"Job {args.job_id} finished with state: {final_state.value}",
            file=sys.stderr,
        )

        if final_state == JobStatus.FAILED:
            sys.exit(1)

        return

    logs = client.jobs.get_logs(
        args.job_id,
        benchmark_index=args.benchmark_index,
        options=options,
    )

    if logs:
        print(logs, end="")


def show_job_results(client, job_id, output_format="table"):
    job = client.jobs.get(job_id)

    if job.effective_state != JobStatus.COMPLETED:
        print(
            f"Warning: job {job_id} is in state "
            f"'{job.effective_state.value}', results may be incomplete.",
            file=sys.stderr,
        )

    if not job.results or not job.results.benchmarks:
        print("No results available.")
        return

    if output_format in ("json", "yaml"):
        data = [b.model_dump(mode="json") for b in job.results.benchmarks]
        output(data, output_format=output_format)
        return

    rows = []

    for benchmark in job.results.benchmarks:
        for metric_name, metric_value in benchmark.metrics.items():
            rows.append(
                {
                    "benchmark": benchmark.id,
                    "provider": benchmark.provider_id,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    if not rows:
        print("No metric results available.")
        return

    output(
        rows,
        output_format=output_format,
        columns=["benchmark", "provider", "metric", "value"],
    )

    if job.results.mlflow_experiment_url:
        print()
        print(f"MLflow experiment: {job.results.mlflow_experiment_url}")


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
    # eval
    # ------------------------------------------------------------------

    eval_parser = subparsers.add_parser(
        "eval",
        help="Submit and manage evaluation jobs",
        description=(
            "Submit and manage evaluation jobs.\n\n"
            "Use 'eval submit' to submit a new evaluation, 'eval status' to track "
            "progress, 'eval results' to fetch outcomes, and 'eval cancel' to "
            "abort a running job."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        required=True,
    )

    # eval status
    status_parser = eval_subparsers.add_parser(
        "status",
        help="Show job status or list all jobs",
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

    # eval submit
    submit_parser = eval_subparsers.add_parser(
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

    # eval cancel
    cancel_parser = eval_subparsers.add_parser(
        "cancel",
        help="Cancel a running or queued evaluation job",
    )

    cancel_parser.add_argument(
        "job_id",
        help="Evaluation job ID",
    )

    cancel_parser.add_argument(
        "--hard-delete",
        action="store_true",
        help="Permanently delete the job instead of cancelling",
    )

    # eval logs
    logs_parser = eval_subparsers.add_parser(
        "logs",
        help="View logs for an evaluation job",
    )

    logs_parser.add_argument(
        "job_id",
        help="Evaluation job ID",
    )

    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Stream logs until the job completes",
    )

    logs_parser.add_argument(
        "--tail",
        type=int,
        default=1000,
        help="Number of lines to show (default: 1000)",
    )

    logs_parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in log output",
    )

    logs_parser.add_argument(
        "--since",
        type=int,
        default=None,
        help="Show logs from the last N seconds",
    )

    logs_parser.add_argument(
        "--benchmark-index",
        type=int,
        default=None,
        help="Show logs for a specific benchmark by index",
    )

    logs_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polls when using --follow (default: 2.0)",
    )

    logs_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop streaming after N seconds (only with --follow)",
    )

    # eval results
    results_parser = eval_subparsers.add_parser(
        "results",
        help="Retrieve and display evaluation results",
    )

    results_parser.add_argument(
        "job_id",
        help="Evaluation job ID",
    )

    results_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json", "yaml", "csv"],
        default="table",
        help="Output format (default: table)",
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

            elif args.command == "eval":
                if args.eval_command == "status":
                    if args.all_jobs and args.job_id:
                        parser.error(
                            "eval status: specify either JOB_ID or --all, not both"
                        )

                    if not args.all_jobs and not args.job_id:
                        parser.error(
                            "eval status: specify JOB_ID or --all"
                        )

                    if args.all_jobs:
                        if args.wait:
                            parser.error(
                                "eval status --all does not support --wait"
                            )

                        show_all_job_status(client)

                    else:
                        show_job_status(
                            client,
                            args.job_id,
                            wait=args.wait,
                            interval=args.interval,
                        )

                elif args.eval_command == "submit":
                    if args.benchmark and not args.provider:
                        parser.error(
                            "eval submit --benchmark requires --provider"
                        )

                    submit(client, args)

                elif args.eval_command == "cancel":
                    cancel_job(
                        client,
                        args.job_id,
                        hard_delete=args.hard_delete,
                    )

                elif args.eval_command == "logs":
                    if args.timeout is not None and not args.follow:
                        parser.error(
                            "eval logs: --timeout requires --follow"
                        )

                    show_job_logs(client, args)

                elif args.eval_command == "results":
                    show_job_results(
                        client,
                        args.job_id,
                        output_format=args.output_format,
                    )

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
