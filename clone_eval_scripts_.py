import argparse
import os
from pathlib import Path

# -------------------------
# Templates
# -------------------------
SE_EVAL_TEMPLATE = "eval/SE_eval.sh"
SINGLE_SEG_TEMPLATE = "eval/single_seg_launch_SE.sh"

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True)
parser.add_argument(
    "--datasets",
    nargs="+",
    required=True,
    help="List of datasets, e.g. TCD_DEMAND TCD_TIMIT_SMALL LRS3_NTCD",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="eval/cloned_scripts",
)
args = parser.parse_args()

run_id = args.run_id
datasets = args.datasets
base_out_dir = Path(args.out_dir) / run_id
base_out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Injected values
# -------------------------
ckpt_path = f"./logs/{run_id}/last.ckpt"

# -------------------------
# Clone functions
# -------------------------
def clone_se_eval(template_path: str, output_path: Path, dataset: str):
    """Patch CKPT_PATH and DATASET in SE_eval.sh"""
    with open(template_path, "r") as f:
        lines = f.read().splitlines()

    ckpt_replaced = False
    dataset_replaced = False
    new_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("CKPT_PATH="):
            new_lines.append(f'CKPT_PATH="{ckpt_path}"')
            ckpt_replaced = True

        elif stripped.startswith("DATASET="):
            new_lines.append(f'DATASET="{dataset}"')
            dataset_replaced = True

        else:
            new_lines.append(line)

    if not ckpt_replaced:
        raise RuntimeError("CKPT_PATH not found in SE_eval template")
    if not dataset_replaced:
        raise RuntimeError("DATASET not found in SE_eval template")

    with open(output_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    os.chmod(output_path, 0o755)
    print(f"[OK] Created {output_path}")


def clone_single_seg(template_path: str, output_path: Path, se_eval_script: Path):
    """Patch the oarsub full_command to call the dataset-specific SE_eval script"""
    with open(template_path, "r") as f:
        lines = f.read().splitlines()

    replaced = False
    new_lines = []

    for line in lines:
        if line.strip().startswith("./eval/SE_eval.sh") and "./eval/SE_eval.sh" in line:
            new_line = line.replace(
                "./eval/SE_eval.sh",
                str(se_eval_script),
            )
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError(
            "SE_eval.sh call not found in single_seg_launch_SE template"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    os.chmod(output_path, 0o755)
    print(f"[OK] Created {output_path}")


# -------------------------
# Main loop: run_id × dataset
# -------------------------
for dataset in datasets:
    se_eval_out = base_out_dir / f"SE_eval_{run_id}_{dataset}.sh"
    single_seg_out = (
        base_out_dir / f"single_seg_launch_SE_{run_id}_{dataset}.sh"
    )

    clone_se_eval(SE_EVAL_TEMPLATE, se_eval_out, dataset)
    clone_single_seg(SINGLE_SEG_TEMPLATE, single_seg_out, se_eval_out)
