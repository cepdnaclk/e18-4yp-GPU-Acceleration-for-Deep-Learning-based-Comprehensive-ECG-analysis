# install all required packages for the project
# can use conda export too. but this is more readable

pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install pandas==2.1.4
pip3 install matplotlib==3.8.2
pip3 install tqdm==4.66.1
pip3 install wfdb==4.1.2
pip3 install wandb==0.16.1
pip3 install einops==0.7.0
pip3 install -U scikit-learn==1.4.1.post1

# after that 
# 'wandb login' 
# then paste your api key

# kepler
#  pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
