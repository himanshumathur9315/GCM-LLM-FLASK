# Step 1: Use an official NVIDIA CUDA image as the base
# This image includes the necessary CUDA drivers and libraries for GPU access.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Step 2: Set up the working environment
WORKDIR /app

# Step 3: Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Step 4: Copy and install Python dependencies
# We copy the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if the requirements file changes.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 5: Copy your application code and model files into the container
COPY . .

# Step 6: Expose the port your Flask app will run on
EXPOSE 5000

# Step 7: Define the command to run your application
# This will execute `python3 app.py` when the container starts.
CMD ["python3", "app.py"]