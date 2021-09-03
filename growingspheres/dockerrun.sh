xhost local:root

sudo docker run \
        -it \
        --rm \
	--env="DISPLAY" \
	--gpus all \
	--shm-size 4G \
	--volume="/etc/group:/etc/group:ro" \
	--volume="/etc/passwd:/etc/passwd:ro" \
	--volume="/etc/shadow:/etc/shadow:ro" \
	--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume "/tmp/.X11-unix:/tmp/.X11-unix:ro" \
	--volume "/dev/snd:/dev/snd" \
        --volume $(pwd):$(pwd) \
        --workdir="$(pwd)" \
       pytorch1.8.0-cuda11.1-cudnn8.runtime:GS_complete
