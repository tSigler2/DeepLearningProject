# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Install Node.js (using a specific version or the latest stable version)
RUN apt-get update && apt-get install -y nodejs npm

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "dsci", "/bin/bash", "-c"]

# Install Python LSP server (and potentially other pip packages)
RUN pip install 'python-lsp-server[all]'
RUN apt-get update && apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic && rm -rf /var/lib/apt/lists/*

# Expose the port JupyterLab will use
EXPOSE 8888

# Start JupyterLab server
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dsci", "jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]