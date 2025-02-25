FROM python:3.10.11

WORKDIR /app

# Copy the requirements from 05_ui
COPY 05_ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the utils folder
COPY utils/ /app/utils/

# Copy the app file from 05_ui
COPY 05_ui/app.py /app/

EXPOSE 9500
CMD ["python", "app.py"]