# install all required packages for the project
# can use conda export too. but this is more readable

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install pandas==2.1.4
pip3 install matplotlib==3.8.2
pip3 install tqdm
pip install wandb

# after that 
# 'wandb login' 
# then paste your api key