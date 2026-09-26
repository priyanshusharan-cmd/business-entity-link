import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
import os
import shutil
import zipfile
from pathlib import Path

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PackageSubmission")


def create_submission_zip(team_name: str = "antigravity_team") -> Path:
    """
    Creates the official submission ZIP archive conforming strictly to contest requirements:
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    """
    zip_filename = config.BASE_DIR / f"{team_name}_submission.zip"
    logger.info("Creating official submission zip package at %s...", zip_filename)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for tsv in ["matching_results.tsv", "candidate_pairs.tsv"]:
            p = config.OUTPUT_DIR / tsv
            if p.exists():
                zf.write(p, f"output/{tsv}")
                logger.info("  + Added output/%s", tsv)

        # 2. Code files
        src_dir = config.BASE_DIR / "src"
        for py_file in src_dir.glob("*.py"):
            zf.write(py_file, f"code/business_entity_resolution/src/{py_file.name}")
            
        req_file = config.BASE_DIR / "requirements.txt"
        if req_file.exists():
            zf.write(req_file, "code/business_entity_resolution/requirements.txt")
            
        readme_file = config.BASE_DIR / "info" / "data" / "student_resource" / "README.md"
        if readme_file.exists():
            zf.write(readme_file, "code/business_entity_resolution/README.md")

        # 3. Documentation template
        doc_file = config.BASE_DIR / "info" / "data" / "student_resource" / "Documentation_template.md"
        if doc_file.exists():
            zf.write(doc_file, "Documentation_template.md")

    logger.info("Submission package ZIP created successfully: %s (%.2f MB)", zip_filename, zip_filename.stat().st_size / (1024*1024))
    return zip_filename


if __name__ == "__main__":
    create_submission_zip()
