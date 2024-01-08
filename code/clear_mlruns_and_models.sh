#!/bin/bash

# Delete mlruns and saved models 

# Prompt the user for confirmation
read -p "Are you sure that you want to delete everything in mlruns and saved_models (This will delete all the training logs and the saved model files)? (YES/NO): " response

# Check the user's response
if [ "$response" = "YES" ] ; then
    # Delete the directories
    rm -rf mlruns
    rm -rf saved_models
    echo "Directories deleted."
else
    echo "Operation canceled."
fi
