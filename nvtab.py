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


def main():

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
    visible_devices = "0,1"      # Delect devices to place workers
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

    demo_dataset_path = '/data/enamine/enamine_parquet/'
    dataset = nvt.Dataset(demo_dataset_path, engine="parquet", part_mem_fraction=0.1)
    ddf = dataset.to_ddf()
    dataset = nvt.Dataset(ddf.head(100))

    lambda_feature = nvt.ColumnGroup(["smiles"])


    new_lambda_feature = lambda_feature >> nvt.ops.LambdaOp(morgan_fingerprint, dependency=["smiles"]) \
                                        >>  nvt.ops.Rename(postfix='_fp')
    processor = nvt.Workflow(new_lambda_feature + 'smiles')

    dataset = processor.fit_transform(dataset)
    print('processor      ', processor)
    print('Type processor ', type(processor), dir(processor))
    print('dataset.to_ddf()', dataset.to_ddf().head())
    print(dataset.num_rows)

from rdkit import Chem
from rdkit.Chem import AllChem
from cddd.inference import InferenceModel


def morgan_fingerprint(col, gdf):
    # m = Chem.MolFromSmiles('CCOC1=CC=C(NC(=O)CN2CCN(C(=O)C(C)OCC3CC3)CC2)C=C1')
    # fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=512)
    # fp = ' '.join(list(fp.ToBitString()))
    # # return fp[:5]
    print(type(col), type(gdf))
    return col

if __name__ == '__main__':
    main()