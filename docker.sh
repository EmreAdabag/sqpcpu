docker run -it \
    -e DISPLAY=$DISPLAY \
    --network host \
    --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
	sqpcpu-env
