# ai-createembeddings
# Install dependencies
pip install -r requirements.txt

# Create embeddings from a text file
python3 create_embeddings.py --model /path/to/your/model.gguf --input yourtext.txt --output embeddings.json

# Run prompts against the embeddings
python3 run_prompt.py --model /path/to/your/model.gguf --embeddings embeddings.json --prompt "Your question here"