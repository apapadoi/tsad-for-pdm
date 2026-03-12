container_ids=$(sudo docker container ls -a --format "{{.ID}}" | head -n 3)

sudo docker container rm $container_ids

sudo docker volume rm pdm-evaluation_mlflow_artifacts

sudo docker volume rm pdm-evaluation_postgres_data