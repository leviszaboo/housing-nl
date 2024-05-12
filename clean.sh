#!/bin/bash

valid_args=("-d" "-a" "-s" "-p1" "-p2")

is_valid_arg() {
    local arg=$1
    for valid_arg in "${valid_args[@]}"; do
        if [[ $arg == "$valid_arg" ]]; then
            return 0
        fi
    done
    return 1
}

if [ "$#" -eq 0 ]; then
    echo "Cleaning output directories..."
    find output -type f -exec rm -f {} \;
    echo "Done!"
    exit 0
fi

delete_data=false
delete_station_data=false
delete_analysis=false
delete_phase_1=false
delete_phase_2=false

for arg in "$@"; do
    if ! is_valid_arg "$arg"; then
        echo "Invalid argument: $arg"
        echo "Usage: ./run.sh -a | -d | -s | -p1 | -p2"
        exit 1
    fi

    case $arg in
        "-d")
            delete_data=true
            ;;
        "-s")
            delete_station_data=true
            ;;
        "-a")
            delete_analysis=true
            ;;
        "-p1")
            delete_phase_1=true
            ;;
        "-p2")
            delete_phase_2=true
            ;;
    esac
done

if [ "$delete_data" = true ]; then
    echo "Deleting main datasets..."
    find output/data -type f ! -name 'stations.csv' -exec rm -f {} \;
fi

if [ "$delete_station_data" = true ]; then
    echo "Deleting station.csv..."
    find output/data -type f -name 'stations.csv' -exec rm -f {} \;
fi

if [ "$delete_analysis" = true ]; then
    echo "Deleting analysis results..."
    find output -path output/data -prune -o -type f -exec rm -f {} \;
fi

if [ "$delete_phase_1" = true ]; then
    echo "Deleting phase 1 results..."
    find output -type f -path '*/phase_1/*' -exec rm -f {} +
fi

if [ "$delete_phase_2" = true ]; then
    echo "Deleting phase 2 results..."
    find output -type f -path '*/phase_2/*' -exec rm -f {} +
fi

echo "Done!"
