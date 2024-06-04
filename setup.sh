# python 3.10 
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install sacrebleu sacrebleu[ko] sacrebleu[ja] transformers==4.40.1 datasets accelerate openai sentencepiece fasttext vllm==0.4.1 protobuf
cd evaluate
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin