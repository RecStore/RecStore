set -x s
cd "$(dirname "$0")"

RECSTORE_PATH="$(cd ".." && pwd)"
Docker_RECSTORE_PATH="/app/RecStore"

sudo docker run --cap-add=SYS_ADMIN --privileged --security-opt seccomp=unconfined  \
--name xmh_recstore --net=host \
-v ${RECSTORE_PATH}:${Docker_RECSTORE_PATH} \
-v /dev/shm:/dev/shm \
-v /dev/hugepages:/dev/hugepages \
-v /dev:/dev -v /nas:/nas \
-w ${Docker_RECSTORE_PATH} --rm -it --gpus all -d recstore:env

