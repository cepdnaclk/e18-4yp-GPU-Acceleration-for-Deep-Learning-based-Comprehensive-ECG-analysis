# upload the code files to ampere server
rsync -avz --exclude='**/*.csv' --exclude='**/*.asc' --exclude='**/*.pyc' --exclude='saved_models/*' --exclude='mlruns/*' -e "ssh -i /home/ishanfdo/.ssh/id_rsa -J e18098@aiken.ce.pdn.ac.lk" * e18098@10.40.18.10:/storage/scratch1/e18-4yp-comp-ecg-analysis
