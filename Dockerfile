# Use an official Python runtime as a parent image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Make port 50051 available to the world outside this container
EXPOSE 50051

# Run the script when the container launches
CMD ["python", "main.py"]

 
# Uncomment code Below for GPU Dockerfile :)

# # Start from a CUDA base image
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
#
# # Set up working directory
# WORKDIR /app
#
# ENV DEBIAN_FRONTEND=noninteractive
#
# # Add deadsnakes PPA for Python 3.11
# RUN apt-get update && \
#     apt-get install -y curl && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update
#
# # Install Python 3.11
# RUN apt-get install -y python3.11 python3.11-distutils python3.11-venv
#
# # Start env
# RUN python3.11 -m venv venv
# ENV PATH=/app/venv/bin:$PATH
#
# # Install pip
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
#     && python3.11 get-pip.py \
#     && rm get-pip.py
#
# # Copy the requirements.txt file
# COPY requirements.txt /app
#
# # Install Python dependencies
# RUN python3.11 -m pip install -r requirements.txt
#
# # Copy your application code to the container
# COPY . /app
#
# # Expose the port the app runs on
# EXPOSE 50051
#
# # Run the script when the container launches
# CMD ["python", "main.py"]
