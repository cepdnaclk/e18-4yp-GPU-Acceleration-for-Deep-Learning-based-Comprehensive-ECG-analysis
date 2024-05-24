# upload the code files to turing server
rsync -avz --exclude='**/*.csv' --exclude='wandb' --exclude='datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977'  --exclude='datasets/PTB_XL_Plus/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1' --exclude='**/*.dat' --exclude='**/*.hea' --exclude='**/__pycache__' --exclude='**/*.atr'  --exclude='**/*.asc' --exclude='**/*.pyc' --exclude='saved_models/*' --exclude='mlruns/*' -e "ssh -i /home/ishanfdo/.ssh/id_rsa -J e18098@aiken.ce.pdn.ac.lk" * e18098@turing.ce.pdn.ac.lk:/storage/scratch/e18-4yp-comp-ecg-analysis
