set -ef -o pipefail

if [ $# -ne 1 ]; then
    echo "Must supply name of container definition, E.G. ubuntu-22.04-base"
    exit 1
fi

NAME=$1

sudo singularity build -F "${NAME}.sif" "${NAME}.def"
