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
sully_data_dir = data_dir / "sully_2022"
economics_data_dir = data_dir / "economics"
geographic_dir = data_dir / "geographic"
gdp_dir = economics_data_dir / "gdp"
tourism_dir = economics_data_dir / "tourism"

# METADATA DIRECTORIES
figures_dir = repo_dir / "figures"


if __name__ == "__main__":
    print(repo_dir)
