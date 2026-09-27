# Disconnected environments

Additional images and model packaging when the cluster has no direct internet access.

* Assuming RHOAI, GPU Operator and dependencies are installed.
* Demo environment has been setup

#### Additional Images

**Note**: The list of additional images for disconnected environment has not been fully verified yet.

| Name | Description |
| :---- | :---- |
| quay.io/rh-aiservices-bu/odh-tec:latest | Open Data Hub Tools |
| registry.redhat.io/ubi9/python-312:latest | Python 3.12 |
| docker.io/amazon/aws-cli:latest | AWS CLI  |
| quay.io/minio/minio | Minio |
| ghcr.io/open-webui/open-webui:main | Open WebUI |
| Value of `RHAIIS_IMAGE` in Makefile (e.g. `registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2`) | vLLM ServingRuntime for CUDA |
| quay.io/ltsai/guidellm:0.3.0 | GuideLLM |
| quay.io/ltsai/benchmark-arena:latest | Model comparison |
| quay.io/modh/vllm@sha256:db766445a1e3455e1bf7d16b008f8946fcbe9f277377af7abb81ae358805e7e2 | RHOAI 2.23 vLLM for cuda |
| quay.io/repository/ltsai/ai-toolkit:latest | Custom AI toolkit. |
| quay.io/ltsai/openshift-nfs-server:latest | NFS server |
| k8s.gcr.io/sig-storage/nfs-subdir-external-provisioner:v4.0.2 | NFS provisioner |

#### Additional Repository

* TBD

#### Download Models

* Download models

```bash
pip3 install --upgrade huggingface_hub
mkdir Qwen/Qwen2.5-VL-7B-Instruct 
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Qwen/Qwen2.5-VL-7B-Instruct 
```

* Package the models into a gzip tarball

```bash
rm -rf Qwen/Qwen2.5-VL-7B-Instruct/.cache Qwen/Qwen2.5-VL-7B-Instruct/.gitattributes
tar --disable-copyfile -cvzf /tmp/Qwen2.5-VL-7B-Instruct.tar.gz Qwen/Qwen2.5-VL-7B-Instruct
tar -tzf /tmp/Qwen2.5-VL-7B-Instruct.tar.gz
```

* You can ignore the warnings if you are doing this from Mac

```text
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
```

* Load into the cluster PVC

```bash
TOOLKIT=$(oc get pods -l app=ai-toolkit -o custom-columns=NAME:.metadata.name --no-headers)

# Verify the tgz layout
cat /tmp/Qwen2.5-VL-7B-Instruct.tar.gz | oc exec -i $TOOLKIT -- tar tzf -

# Unpack to the pod's /mnt/models
cat /tmp/Qwen2.5-VL-7B-Instruct.tar.gz | oc exec -i $TOOLKIT  -- tar xzf - -C /mnt/models
```

* Verify the model path on the PVC

```bash
oc exec $TOOLKIT -- ls -la /mnt/models
```

#### Open WebUI

* Enable offline mode

```bash
oc set env deploy/open-webui OFFLINE_MODE=True
```

* Download embedding models

```bash
OPEN_WEBUI=$(oc get pods -l app=open-webui -o custom-columns=NAME:.metadata.name --no-headers)
```

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --cache-dir .

oc rsh $OPEN_WEBUI /usr/bin/mkdir -p \
/app/backend/data/cache/embedding/models

oc cp models--sentence-transformers--all-MiniLM-L6-v2 \
$OPEN_WEBUI:/app/backend/data/cache/embedding/models
```

```bash
hf download Systran/faster-whisper-base --cache-dir .

oc rsh $OPEN_WEBUI /usr/bin/mkdir -p \
/app/backend/data/cache/whisper/models

oc cp models--Systran--faster-whisper-base $OPEN_WEBUI:/app/backend/data/cache/whisper/models/
```

* Restart the pod. You can also [reset](appendix.md#configure-open-webui-for-multiple-endpoints) Open WebUI.

```bash
oc delete pod $OPEN_WEBUI
```

