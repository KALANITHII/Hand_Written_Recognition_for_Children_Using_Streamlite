# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Install required system packages and mpg123
RUN apt-get update -y && apt-get install -y mpg123

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
