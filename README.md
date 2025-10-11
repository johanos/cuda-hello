Simple CUDA program to add two vectors.

# how to run with docker

```bash
TAG_NAME=cuda-hello:dev
docker build -t $TAG_NAME .
docker run --rm -it --gpus all $TAG_NAME
```
