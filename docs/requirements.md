# Requirements

Hardware, software, and storage expectations for this bootstrap environment.

* x86 architecture (AMD GPUs are also supported in many flows)
* NVIDIA GPU recommended for the examples in this repo

### Single Node OpenShift

| Type | Qty | vCPU | Memory (GB) | Disk (GB) |
| :---- | :---- | :---- | :---- | :---- |
| SNO | 1 | 32 | 64 | 200 |

### OpenShift Cluster

| Type | Qty | vCPU | Memory (GB) | Disk (GB) |
| :---- | :---- | :---- | :---- | :---- |
| Control plane | 1 | 8 | 16 | 200 |
| CPU Worker | 2 | 16 | 32 | 200 |

### Nvidia GPU

* Ada Lovelace: L4, L20, L40S (no MIG, no NVLink)  
* Ampere: A100  (\>=A40 for MIG)  
* Hopper: H20, H100, H200  
* Blackwell: B200

| GPU Model | INT4 | FP8 | FP16 | BF16 |
| :---- | :---- | :---- | :---- | :---- |
| L4 | Supported | Supported | Supported | Supported |
| L20 | Supported | Supported | Supported | Supported |
| L40s | Supported | Supported | Supported | Supported |
| A100 | Supported | Supported | Supported | Supported |
| H20 | Supported | Supported | Supported | Supported |
| H100 | Supported | Supported | Supported | Supported |
| H200 | Supported | Supported | Supported | Supported |
| B200 | Not Supported | Supported | Supported | Supported |

* It is recommended to deploy additional GPU worker node(s) with 1 or more GPU.
* The quantity and type of GPU depends on the type of tests being done. E.g.
  * Model size
  * Number of models
  * Benchmarking
  * Number of GPU required (tensor parallelism)
  * NVLink required
  * Concurrency test (RPS)

This demo has been done on AWS using:

* g6.12xlarge: x4 Nvidia L4
* g6e.12xlarge: x4 Nvidia L40S

On AWS, use `scripts/clone-machineset.sh` to clone a worker MachineSet to a GPU instance type (new MachineSet starts at `replicas=0`). Not every availability zone offers every GPU instance type.

```bash
scripts/clone-machineset.sh --help
scripts/clone-machineset.sh --list
scripts/clone-machineset.sh g6.4xlarge --dry-run
scripts/clone-machineset.sh ocp-example-worker-ap-northeast-1a p4d.24xlarge --on-demand
```

### Software

| Software Description | Version |
| :---- | :---- |
| Internet/Proxy access to HF, quay.io and Red Hat registry | N/A |
| Red Hat OpenShift Container Platform | Tested on 4.21 |
| Red Hat OpenShift AI | Tested on 3.5 (early access) |
| vLLM ServingRuntime image | See `RHAIIS_IMAGE` in Makefile |
| Nvidia GPU Operator | Latest |
| NFD Operator | Latest |

### CSI Storage

* CSI is available for RWO. NFS provisioner will be used to deploy RWX.
* For SNO, additional disk is required for logical volume manager.
* Min 300GB

