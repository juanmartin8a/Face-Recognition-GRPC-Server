# Start from a CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set up working directory
WORKDIR /app

# Add deadsnakes PPA for Python 3.11
RUN apt update && \
    apt install -y curl && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update

# Install Python 3.11
RUN apt install -y python3.11 python3.11-distutils python3.11-venv

# Start env
RUN python3.11 -m venv venv
ENV PATH=/app/venv/bin:$PATH
ENV ENV=prod

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Copy the requirements.txt file
COPY requirements.txt /app

# Install Python dependencies
RUN python3.11 -m pip install -r requirements.txt

# Copy your application code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 50051

# Command to run the application
CMD ["python3.11", "main.py"]
