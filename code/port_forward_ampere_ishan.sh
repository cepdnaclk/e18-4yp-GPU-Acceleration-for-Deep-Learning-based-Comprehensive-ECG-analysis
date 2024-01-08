# forward the port 5000 to view the mlflow UI
ssh -i /home/ishanfdo/.ssh/id_rsa -L 5000:localhost:5000 -J e18098@aiken.ce.pdn.ac.lk -N -f e18098@10.40.18.10
