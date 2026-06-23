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

    # Fallback for environments where `git -C` cannot resolve OneDrive-backed
    # paths but the repository checkout is still present.
    for parent in (file_dir, *file_dir.parents):
        if (parent / ".git").exists():
            return parent

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
sully_og_dir = data_dir / "sully_og"
economics_data_dir = data_dir / "economics"
geographic_dir = data_dir / "geographic"
gdp_dir = economics_data_dir / "gdp"
tourism_dir = economics_data_dir / "tourism"
coastlines_dir = data_dir / "coastlines"
sentinel_coast_dir = coastlines_dir / "S2Coast2023_ShapeFile_vector"
meow_dir = data_dir / "MEOW"
mpas_dir = data_dir / "mpas" / "WDPA_Jun2026_Public_shp"
diversity_dir = data_dir / "ecoregion_diversity"
env_dir = data_dir / "env_vars"
reef_check_dir = data_dir / "reef_check"

# METADATA DIRECTORIES
figures_dir = repo_dir / "figures"


if __name__ == "__main__":
    print(repo_dir)
