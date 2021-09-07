# set base image (host OS)
FROM python:3.8-buster


# set the working directory in the container
WORKDIR /usr/src/ser

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY *.joblib .
COPY predict.py .

# install dependencies & upgrade
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY ser ./ser
COPY api ./api

COPY /Users/pankajpatel/.gcppankaj/wagon-bootcamp-322818-752a3eb15b98.json /wagon-bootcamp-322818-752a3eb15b98.json

# command to run on container start
# load web server with code autoreload
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

# CMD streamlit run app.py  --server.port 8080
