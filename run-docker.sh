sudo docker run --name documentembedding \
	-p 5500:8080 \
	-v "/models/hf:/models/hf" \
  -e HF_HOME=/models/hf \
	documentembedding \
  &