# gemmagpt
GemmaGPT là mô hình ngôn ngữ lớn sử dụng những kỹ thuật tương tự như Gemini nhưng nhẹ hơn.

## Try it out with PyTorch

Prerequisite: make sure you have setup docker permission properly as a non-root user.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Build the docker image.

```bash
DOCKER_URI=gemma:${USER}

docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}
```

### Run Gemma inference on CPU.

```bash
PROMPT="The meaning of life is"

docker run -t --rm \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}"
    # add `--quant` for the int8 quantized model.

## Dùng tool pycharm thiết lập môi trường ảo bằng conda3 với Python 3.10.13 và pytorch == 2.1.2 rồi chạy lệnh ở dưới.
## main.py được clone từ scripts/run.py
python main.py --ckpt=/home/nghiatc/lab/labGemma/pytorch-2b-it/gemma-2b-it.ckpt --variant=2b --output_len=300 --prompt="The meaning of life is"
```

### Run Gemma inference on GPU.

```bash
PROMPT="The meaning of life is"

docker run -t --rm \
    --gpus all \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --device=cuda \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    --prompt="${PROMPT}"
    # add `--quant` for the int8 quantized model.
```

## Try It out with PyTorch/XLA

### Build the docker image (CPU, TPU).

```bash
DOCKER_URI=gemma_xla:${USER}

docker build -f docker/xla.Dockerfile ./ -t ${DOCKER_URI}
```

### Build the docker image (GPU).

```bash
DOCKER_URI=gemma_xla_gpu:${USER}

docker build -f docker/xla_gpu.Dockerfile ./ -t ${DOCKER_URI}
```

### Run Gemma inference on CPU.

```bash
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=CPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on TPU.

Note: be sure to use the docker container built from `xla.Dockerfile`.

```bash
docker run -t --rm \
    --shm-size 4gb \
    -e PJRT_DEVICE=TPU \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```

### Run Gemma inference on GPU.

Note: be sure to use the docker container built from `xla_gpu.Dockerfile`.

```bash
docker run -t --rm --privileged \
    --shm-size=16g --net=host --gpus all \
    -e USE_CUDA=1 \
    -e PJRT_DEVICE=CUDA \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run_xla.py \
    --ckpt=/tmp/ckpt \
    --variant="${VARIANT}" \
    # add `--quant` for the int8 quantized model.
```
