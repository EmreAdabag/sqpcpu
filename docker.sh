docker run -it \
    -e DISPLAY=$DISPLAY \
    -e ROS_DOMAIN_ID=0 \
    --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
	sqpcpu-env

