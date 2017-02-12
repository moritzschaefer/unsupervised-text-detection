docker build -t uns .
docker run -it --name bashed-uns -v /home/mlp/models:/usr/src/app/data uns
