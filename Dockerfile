FROM ros:humble-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    ros-humble-urdfdom \
    ros-humble-hpp-fcl \
    ros-humble-urdfdom-headers \
    liburdfdom-headers-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11
RUN pip3 install pybind11

# Install OSQP and OsqpEigen
RUN git clone --recursive https://github.com/osqp/osqp && \
    cd osqp && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" .. && \
    cmake --build . && \
    cmake --install . && \
    cd ../.. && \
    rm -rf osqp

RUN git clone https://github.com/robotology/osqp-eigen.git && \
    cd osqp-eigen && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf osqp-eigen

# built pinocchio from source
RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio && \
    cd pinocchio && \
    git checkout master && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DCMAKE_PREFIX_PATH="/opt/ros/humble;/usr/local" && \
    make -j4 && \
    make install

# Set environment variables, add to bashrc
RUN echo "export PATH=/usr/local/bin:$PATH" >> ~/.bashrc
RUN echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN echo "export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH" >> ~/.bashrc
RUN echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.bashrc

# Install Pinocchio from ROS packages
# RUN apt-get update && apt-get install -y \
#     ros-humble-pinocchio \
#     ros-humble-hpp-fcl \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Source ROS environment in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc