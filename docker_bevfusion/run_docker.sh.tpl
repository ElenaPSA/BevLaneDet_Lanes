UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)

nvidia-docker run -it -d \
       -v/mnt/data3:/data3 \
       -v /mnt/data:/data \
       -v $PWD:/workspace \
       -w /workspace \
       -e "HOME=/workspace" \
       -p 3170-3175:3170-3175 \
       --name bevfusion_$UNAME \
       --shm-size="256g" \
       bevfusion:latest
