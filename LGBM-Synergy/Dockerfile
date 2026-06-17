# Usamos una imagen oficial de Miniconda sobre Linux
FROM continuumio/miniconda3:latest

# Directorio de trabajo dentro del contenedor
WORKDIR /usr/src/app

# Copiamos el archivo de entorno original de Linux
COPY lgbm_environment.yml .

# Creamos el entorno de conda tal cual viene en el repositorio
RUN conda env create -f lgbm_environment.yml

# Activamos el entorno lgbm inyectándolo en el PATH del contenedor
ENV PATH /opt/conda/envs/lgbm/bin:$PATH

# Copiamos todo el código del repositorio al contenedor
COPY . .

# Al arrancar, nos da una terminal Linux lista para trabajar
CMD ["bash"]