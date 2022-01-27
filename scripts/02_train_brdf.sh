gpus='0'

# I. Learning BRDF Priors (training and validation)
repo_dir="/home/v-xiuli1/workspace/nerfactor"
data_root="/home/v-xiuli1/databag/merl/brdfs_npz/ims512_envmaph16_spp1"
outroot="/home/v-xiuli1/weights/brdf"
viewer_prefix='' # or just use ''
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='brdf.ini' --config_override="data_root=$data_root,outroot=$outroot,viewer_prefix=$viewer_prefix"

# # II. Exploring the Learned Space (validation and testing)
# ckpt="$outroot/lr1e-2/checkpoints/ckpt-50"
# REPO_DIR="$repo_dir" "$repo_dir/nerfactor/explore_brdf_space_run.sh" "$gpus" --ckpt="$ckpt"