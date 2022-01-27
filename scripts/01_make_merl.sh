home_root='/home/v-xiuli1'
repo_dir="$home_root/workspace/nerfactor"
indir="$home_root/databag/merl/brdfs"
ims='512'
outdir="$home_root/databag/merl/brdfs_npz/ims${ims}_envmaph16_spp1"
REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"