# general
import subprocess
from pathlib import Path


def get_repo_root():
    # Get the directory where this file is located
    file_dir = Path(__file__).parent.resolve()

    # Run 'git rev-parse --show-toplevel' from the file's directory
    # to get the root directory of the Git repository containing this file
    git_root = subprocess.run(
        ["git", "-C", str(file_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if git_root.returncode == 0:
        return Path(git_root.stdout.strip())
    else:
        raise RuntimeError("Unable to determine Git repository root directory.")


"""
Defines globals used throughout the codebase.
"""

###############################################################################
# Folder structure naming system
###############################################################################

# REPO DIRECTORIES
repo_dir = get_repo_root()
data_dir = repo_dir / "data"

# DATA DIRECTORIES
# Note: data might be in sully_hbb or sully_2022, depending on setup
sully_hbb_data_dir = data_dir / "sully_hbb"  # Check if exists, otherwise use sully_2022
if not sully_hbb_data_dir.exists():
    sully_hbb_data_dir = data_dir / "sully_2022"


# METADATA DIRECTORIES
figures_dir = repo_dir / "figures"


if __name__ == "__main__":
    print(repo_dir)
