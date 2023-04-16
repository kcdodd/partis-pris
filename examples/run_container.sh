set -ef -o pipefail

if [ $# -ne 1 ]; then
    echo "Must supply name container, E.G. ubuntu-22.04-base"
    exit 1
fi

# must create mod.img at least once to save modifications E.G. "bash make_overlay.sh mod 500"

NAME=$1
VENV=$PWD/venv

#module load singularity
if [ ! -d "$VENV" ]; then
  mkdir "$VENV"
fi

singularity run --app venv --cleanenv --no-home --bind "$VENV:/venv" "${NAME}.sif"

# singularity shell --cleanenv --no-home --overlay mod.img "${NAME}.sif"
