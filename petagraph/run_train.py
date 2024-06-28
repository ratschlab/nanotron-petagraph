"""
Nanotron training script example using a custom dataloader.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=2 examples/custom-dataloader/run_train.py --config-file examples/custom-dataloader/config_custom_dl.yaml
```
"""
import argparse
from typing import Dict, cast
from pathlib import Path
from tqdm import tqdm

import numpy as np
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.dataloader import (
    DataCollatorForCLM,
    clm_process,
    get_dataloader_worker_init,
    get_datasets,
    get_train_dataloader,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

from nanotron.data.petagraph_dataset import PetaGraphStreamDataset

logger = logging.get_logger(__name__)


class EmptyInfiniteDataset:
    """Hack as removing all columns from a datasets.Dataset makes the number of rows 0."""

    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, item) -> Dict:
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        return self._length


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: custom data generator
    if data.dataset is None:

        log_rank("Using custom data generator", logger=logger, level=logging.INFO, rank=0)

        # Configure vocabulary
        VOCABULARY = {
            "BOS": 0, "EOS": 1, "PAD": 2, "UNK": 3,
            "A": 4, "C": 5, "G": 6, "T": 7
        }
        bos_config_id = trainer.config.model.model_config.bos_token_id
        eos_config_id = trainer.config.model.model_config.eos_token_id
        pad_config_id = trainer.config.model.model_config.pad_token_id

        assert len(VOCABULARY) == trainer.config.model.model_config.vocab_size, "Vocabulary size mismatch"
        assert VOCABULARY["BOS"] == bos_config_id, "BOS token ID mismatch"
        assert VOCABULARY["EOS"] == eos_config_id, "EOS token ID mismatch"
        assert VOCABULARY["PAD"] == pad_config_id, "PAD token ID mismatch"

        # Load the sequence file candidates
        sequence_files_path = Path(data.sequence_files_path)
        log_rank(f"Loading sequence files from {sequence_files_path}", logger=logger, level=logging.INFO, rank=0)

        # Load files with lines
        if "unitig" in sequence_files_path.name:
            load_type = "unitig"
            url_format = "s3://logan-pub/u/{accession}/{accession}.unitigs.fa.zst"
        elif "contig" in sequence_files_path.name:
            load_type = "contig"
            url_format = "s3://logan-pub/c/{accession}/{accession}.contigs.fa.zst"
        else:
            raise ValueError("Data path must contain either 'unitig' or 'contig'")
        
        # Load data from text file
        all_files = []
        for line in sequence_files_path.open():
            accession = line.strip().strip("\n")
            url = url_format.format(accession=accession)
            all_files.append(url)
        log_rank(f"Found {len(all_files)} {load_type} files", logger=logger, level=logging.INFO, rank=0)

        # TODO: if resuming from a checkpoint, we need to skip the already consumed files
        if consumed_train_samples > 0:
            raise NotImplementedError("Resuming from a checkpoint is not yet supported")

        # Compute size and rank of dataloader workers
        dp_ranks_size = trainer.parallel_context.dp_pg.size()
        dp_rank = trainer.parallel_context.dp_pg.rank()
        log_rank(f"Splitting files across {dp_ranks_size} data parallel ranks", logger=logger, level=logging.INFO, rank=0)

        # We do not use a sampler and handle data parallelism at the file level
        # We need to split the files between the ranks
        files_per_rank = len(all_files) // dp_ranks_size
        start_idx = dp_rank * files_per_rank
        end_idx = start_idx + files_per_rank
        if dp_rank == dp_ranks_size - 1:
            end_idx = len(all_files)

        train_sequence_files = all_files[start_idx:end_idx]
        for r in range(dp_ranks_size):
            log_rank(f"Rank {r} has {len(train_sequence_files)} train files, eg.: {train_sequence_files[:2]}",
                     logger=logger, level=logging.INFO, rank=r)

        # Set or read from config dataloader workers
        num_dl_workers = 0

        # If we are using pipeline parallelism, then we use the same approach
        # as in dataloader.get_train_dataloader and create a dummy dataset on the 
        # ranks, which are not part of input or output pipeline parallel ranks.
        pp_ranks_size = trainer.parallel_context.pp_pg.size()
        if pp_ranks_size > 1 and dist.get_rank(trainer.parallel_context.pp_pg) not in [
            input_pp_rank, output_pp_rank,
        ]:
            dataset_length = len(train_sequence_files) * 100_000
            train_dataset = EmptyInfiniteDataset(length=dataset_length)
            # No need to spawn a lot of workers, we can just use main
            num_dl_workers = 0

        else:
            rank = dist.get_global_rank()
            train_dataset = PetaGraphStreamDataset(
                logger=logger,
                url_list=train_sequence_files,
                vocabulary=VOCABULARY,
                from_cloud=True, # not mock_data,
                maxlen=trainer.sequence_length + 1,
                create_attention_mask=True,
                prefetch_sequences=data.prefetch_buffer_seq_size,
                log_directory=trainer.config.checkpoints.checkpoints_path
                rank=rank
            )


        data_collator = DataCollatorForCLM(
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=trainer.parallel_context,
            padding_index=pad_config_id,
            unknown_index=VOCABULARY["UNK"],
        )

        log_rank(f"Using {num_dl_workers} dataloader workers", logger=logger, level=logging.INFO, rank=0)

        return DataLoader(
            train_dataset,
            batch_size=trainer.micro_batch_size,
            collate_fn=data_collator,
            drop_last=True,
            num_workers=num_dl_workers,
            pin_memory=True,
            worker_init_fn=get_dataloader_worker_init(dp_rank=trainer.parallel_context.dp_pg.rank()),
        )
    
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
