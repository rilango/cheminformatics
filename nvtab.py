import os
import glob
import shutil

import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size


# Choose a "fast" root directory for this example
BASE_DIR = os.environ.get("BASE_DIR", "./basedir")

# Define and clean our worker/output directories
dask_workdir = os.path.join(BASE_DIR, "workdir")
demo_output_path = os.path.join(BASE_DIR, "demo_output")
demo_dataset_path = os.path.join(BASE_DIR, "demo_dataset")

# Ensure BASE_DIR exists
if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)

# Make sure we have a clean worker space for Dask
if os.path.isdir(dask_workdir):
    shutil.rmtree(dask_workdir)
os.mkdir(dask_workdir)

# Make sure we have a clean output path
if os.path.isdir(demo_output_path):
    shutil.rmtree(demo_output_path)
os.mkdir(demo_output_path)

# Get device memory capacity
capacity = device_mem_size(kind="total")

# Deploy a Single-Machine Multi-GPU Cluster
protocol = "tcp"             # "tcp" or "ucx"
visible_devices = "0,1,2,3"  # Delect devices to place workers
device_spill_frac = 0.9      # Spill GPU-Worker memory to host at this limit.
                             # Reduce if spilling fails to prevent
                             # device memory errors.
cluster = None               # (Optional) Specify existing scheduler port
if cluster is None:
    cluster = LocalCUDACluster(
        protocol = protocol,
        CUDA_VISIBLE_DEVICES = visible_devices,
        local_directory = dask_workdir,
        device_memory_limit = capacity * device_spill_frac,
    )

# Create the distributed client
client = Client(cluster)
client

# Initialize RMM pool on ALL workers
def _rmm_pool():
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=None, # Use default size
    )

client.run(_rmm_pool)

# Write a "largish" dataset (~20GB).
# Change `write_count` and/or `freq` for larger or smaller dataset.
# Avoid re-writing dataset if it already exists.
write_count = 25
freq = "1s"
if not os.path.exists(demo_dataset_path):

    def _make_df(freq, i):
        df = cudf.datasets.timeseries(
            start="2000-01-01", end="2000-12-31", freq=freq, seed=i
        ).reset_index(drop=False)
        df["name"] = df["name"].astype("object")
        df["label"] = cp.random.choice(cp.array([0, 1], dtype="uint8"), len(df))
        return df

    dfs = [delayed(_make_df)(freq, i) for i in range(write_count)]
    dask_cudf.from_delayed(dfs).to_parquet(demo_dataset_path, write_index=False)
    del dfs


# Create a Dataset
# (`engine` argument optional if file names appended with `csv` or `parquet`)
ds = nvt.Dataset(demo_dataset_path, engine="parquet", part_size="500MB")
ds.to_ddf().head()



# Example of global shuffling outside an NVT Workflow
ddf = ds.to_ddf().shuffle("id", ignore_index=True)
ds = nvt.Dataset(ddf)
ds.to_ddf()


del ds
del ddf

dataset = nvt.Dataset(demo_dataset_path, engine="parquet", part_mem_fraction=0.1)


cat_features = ["name", "id"] >> ops.Categorify(
    out_path=demo_output_path,  # Path to write unique values used for encoding
)
cont_features = ["x", "y"] >> ops.Normalize()

workflow = nvt.Workflow(cat_features + cont_features + ["label", "timestamp"], client=client)




shuffle = Shuffle.PER_WORKER  # Shuffle algorithm
out_files_per_proc = 8        # Number of output files per worker
workflow.fit_transform(dataset).to_parquet(
    output_path=os.path.join(demo_output_path,"processed"),
    shuffle=shuffle,
    out_files_per_proc=out_files_per_proc,
)



dask_cudf.read_parquet(os.path.join(demo_output_path,"processed")).head()




ddf = workflow.transform(dataset).to_ddf()
ddf = ddf.groupby(["name"]).max() # Optional follow-up processing
ddf.to_parquet(os.path.join(demo_output_path, "dask_output"), write_index=False)



dask_cudf.read_parquet(os.path.join(demo_output_path, "dask_output")).compute()


