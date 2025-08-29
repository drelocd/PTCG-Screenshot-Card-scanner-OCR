Automation wont work with ocr so this is an agent.  
Fully-Self hosted   
Download (4-bit quantized GGUF model)- git clone https://github.com/haotian-liu/LLaVA  
cd LLaVA  
pip install -e .  
docker run -d --name n8n -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n  
