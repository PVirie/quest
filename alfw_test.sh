source .venv/bin/activate

# check if alfworld-play-tw is installed
if ! command -v alfworld-play-tw &> /dev/null; then
    echo "installing alfworld..."
    pip install alfworld
    alfworld-download
fi

alfworld-play-tw
