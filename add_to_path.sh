
segway_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $segway_dir

# check if path is correct
if [[ $PYTHONPATH != *${segway_dir}* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=${segway_dir}:$PYTHONPATH
fi
