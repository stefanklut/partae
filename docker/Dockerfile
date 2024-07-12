FROM condaforge/mambaforge

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get clean -y && \
    apt-get install -y \
    apt-utils git \
    ninja-build gcc g++ \
    ffmpeg libsm6 libxext6 libegl1 libopengl0 && \
    apt-get clean -y && \
    apt-get autoremove -y

WORKDIR /src/
COPY separation/ separation/
COPY _entrypoint.sh /_entrypoint.sh

WORKDIR /src/separation
# Removing the nsight-compute folder as it is not needed for runtime
RUN mamba env create -f environment.yml && \
    mamba clean -y --all --force-pkgs-dirs
# rm -r /opt/conda/envs/separation/nsiglht* 

# SHELL ["conda", "run", "-n", "separation", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/separation/bin:$PATH
ENV CONDA_DEFAULT_ENV=separation
ENV ENV_NAME=separation

# Fix cuda permissions with the docker
# RUN chmod go+rx /opt/conda/envs/separation/pkgs/cuda-toolkit/
# Fix git permissions with the docker
# RUN git config --system --add safe.directory /src/separation

ENTRYPOINT ["/_entrypoint.sh"]
CMD ["/bin/bash"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "separation", "/bin/bash", "-c"]

USER 1000
