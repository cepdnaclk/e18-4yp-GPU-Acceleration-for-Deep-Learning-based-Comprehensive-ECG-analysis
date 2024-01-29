# upload the code files to ampere server
rsync -avz --exclude='**/*.csv' --exclude='wandb' --exclude='**/*.dat' --exclude='**/*.hea' --exclude='**/*.atr' --exclude='**/*.asc' --exclude='**/*.pyc' --exclude='saved_models/*' --exclude='mlruns/*' -e "ssh -J e18100@aiken.ce.pdn.ac.lk" * e18100@10.40.18.10:/storage/scratch1/e18-4yp-comp-ecg-analysis
