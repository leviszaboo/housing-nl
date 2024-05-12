#!/bin/bash
valid_args=("--analysis_only" "--dataset_only" "--skip_station_data" "--phase_1" "--phase_2")

is_valid_arg() {
    local arg=$1
    for valid_arg in "${valid_args[@]}"; do
        if [[ $arg == "$valid_arg" ]]; then
            return 0
        fi
    done
    return 1
}

for arg in "$@"; do
    if ! is_valid_arg "$arg"; then
        echo "Invalid argument: $arg"
        echo "Usage: ./run.sh --analysis_only | --dataset_only | --skip_station_data"
        exit 1
    fi
done

if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d .venv ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
        echo "Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

echo "Checking dependencies..."
pip3 install -r requirements.txt | grep -v 'already satisfied'

python3 main.py "$@"

echo "Compressing files in output/maps..."
find output/maps -type f -name '*.gz' -exec rm -f {} \;
find output/maps -type f -name '*.html' -exec gzip {} \;

echo "Done!"

