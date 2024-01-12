# upload the code files to turing server
rsync -avz --exclude='**/*.csv' --exclude='wandb' --exclude='**/*.dat' --exclude='**/*.hea' --exclude='**/*.atr'  --exclude='**/*.asc' --exclude='**/*.pyc' --exclude='saved_models/*' --exclude='mlruns/*' -e "ssh -i /home/ishanfdo/.ssh/id_rsa -J e18098@aiken.ce.pdn.ac.lk" * e18098@turing.ce.pdn.ac.lk:/storage/scratch/e18-4yp-comp-ecg-analysis
