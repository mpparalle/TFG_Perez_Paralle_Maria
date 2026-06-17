# Usamos la imagen base de PyTorch que ya está disponible en local
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Directorio de trabajo
WORKDIR /usr/src/app

# Instalamos dependencias científicas y de visualización vía pip
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    seaborn \
    rdkit \
    networkx \
    pyyaml \
    matplotlib

# Instalamos PyTorch Geometric y sus dependencias binarias compiladas (torch-scatter, torch-sparse)
# utilizando las ruedas (wheels) oficiales preparadas para PyTorch 2.0.1 y CUDA 11.7
RUN pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Inyectamos el directorio de IMPROVE_repo en el PYTHONPATH
ENV PYTHONPATH /usr/src/IMPROVE_repo

# Copiamos todo el código de DeepDDS al contenedor
COPY . .

# Al arrancar, iniciamos bash
CMD ["bash"]
