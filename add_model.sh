#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <modelfile_name>"
    exit 1
fi
if [ ! -f "$2" ]; then
    echo "Model file '$2' does not exist."
    exit 1
fi

docker cp "$2" ollama:/tmp/"$2"
docker exec -it ollama ollama create $1 -f /tmp/"$2"
