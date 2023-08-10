# Use an official Python base image from the Docker Hub
FROM python:3.10-slim

# Install browsers
RUN apt-get update && apt-get install -y \
    ca-certificates \ 
    xvfb

# Install utilities
RUN apt-get install -y curl jq wget git

# Declare working directory
WORKDIR /app

# Copy the current directory contents into the Workspace.
COPY . /app

# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
