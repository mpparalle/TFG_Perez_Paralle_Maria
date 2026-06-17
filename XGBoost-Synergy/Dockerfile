# Official Miniconda base image
FROM continuumio/miniconda3:latest

# Working directory inside the container
WORKDIR /usr/src/app

# Copy the environment file
COPY xgboostsynergy_environment.yml .

# Create the conda environment
RUN conda env create -f xgboostsynergy_environment.yml

# Activate the conda environment in PATH
ENV PATH /opt/conda/envs/xgboostdrp/bin:$PATH

# Set the PYTHONPATH to include the IMPROVE library
ENV PYTHONPATH /usr/src/IMPROVE_repo

# Copy the model code into the container
COPY . .

# Run bash on start
CMD ["bash"]
