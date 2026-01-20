# https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration
#

docker build -t splat .

docker run -it \
--name splat-test \
--gpus all \
--volume ~/data:/mnt/data \
--volume ~/output:/mnt/output \
splat

docker start splat-test -ai bash
docker start splat-test -ai bash

docker cp ./test.sh splat-test:/workspace/gaussian-splatting-reflection
