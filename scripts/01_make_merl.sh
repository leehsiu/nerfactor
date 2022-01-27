proj_root='/home/xiuli'
repo_dir="$proj_root/nerfactor"
indir="$proj_root/data/merl/brdfs"
ims='512'
outdir="$proj_root/data/brdf_merl_npz/ims${ims}_envmaph16_spp1"
REPO_DIR="$repo_dir" "$repo_dir"/data_gen/merl/make_dataset_run.sh "$indir" "$ims" "$outdir"