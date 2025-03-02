# syntax=docker/dockerfile:1

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7
FROM python:3.10.11

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Copy the requirements file from 05_ui
COPY 05_ui/requirements.txt /app/requirements.txt

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the utils folder into the container.
COPY 05_ui/utils /app/utils

COPY 05_ui/app.py /app/app.py

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
