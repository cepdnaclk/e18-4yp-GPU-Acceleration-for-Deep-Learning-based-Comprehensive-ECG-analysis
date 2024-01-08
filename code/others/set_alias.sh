# set an alias '4yp' to the 4YP directory
# add this to .bashrc to make it permanent
# or run 'source set_alias.sh' to make it temporary

# Get the current hostname
hostname=$(hostname)

# Check if the hostname is 'turing'
if [ "$hostname" == "turing" ]; then
    alias 4yp='cd /storage/scratch/e18-4yp-comp-ecg-analysis'
# Check if the hostname is 'ampere'
elif [ "$hostname" == "ampere" ]; then
    alias 4yp='cd /storage/scratch1/e18-4yp-comp-ecg-analysis'
fi
