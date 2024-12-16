ARG base

FROM $base

ARG threads
ARG cuda

# set a directory for the app
WORKDIR /DiFfRG/

# copy the git to the container
COPY ./.git ./.git

# install dependencies
RUN dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex tbb-devel python3 python3-pip gsl-devel
SHELL [ "/usr/bin/scl", "enable", "gcc-toolset-12"]
RUN git reset --hard
RUN bash -i build.sh -j $threads -f $cuda -i /opt/DiFfRG

# define the port number the container should expose
EXPOSE 5000

# run the command
CMD ["scl enable gcc-toolset-12 bash"]
