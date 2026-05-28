from dataset.ghosts.create_ghosts import create_ghosts, GHOSTS_PATH

def load_ghosts():
    create_ghosts()

    with open(GHOSTS_PATH, "r", encoding="utf-8") as f:
        ghosts = [line.strip() for line in f if line.strip()]

    return ghosts