# =============================================================================
# 
# Copyright (C) 2024, Manuel Burger
# 
# Petagraph Dataset
#
# =============================================================================
from torchdata.datapipes.iter import IterableWrapper, S3FileLoader, \
    FileOpener, Mapper, StreamReader, FSSpecFileOpener, Prefetcher

from functools import partial
import logging
import torch
import random
from tqdm import tqdm
import zstd
import numpy as np
from typing import Dict, Optional, Tuple

from pathlib import Path
from Bio import SeqIO
from io import StringIO

from nanotron.logging import log_rank


# =============================================================================
# Utility functions
# =============================================================================
def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


# =============================================================================
# Dataset class
# =============================================================================
class PetaGraphStreamDataset(torch.utils.data.IterableDataset):

    def __init__(self, 
        logger,
        url_list: list[str],
        vocabulary: dict[str, int],
        from_cloud: bool = False,
        maxlen: int = 128,
        samples_per_epoch: int = -1,
        create_attention_mask: bool = True,
        debug: bool = False,
        prefetch_sequences: int = 4096,
        prefetch_decompressed_files: int = 10,
        prefetch_fasta_parsing: int = 10,
    ):

        self.samples_per_epoch = samples_per_epoch
        self.maxlen = maxlen
        self.create_attention_mask = create_attention_mask
        self.debug = debug

        self.logger = logger
        self.logging_func = partial(log_rank, logger=logger, level=logging.INFO, rank=0)
        self.logging_func("=====================================")
        self.logging_func(f"Creating PetaGraphStreamDataset with maxlen {maxlen}")
        self.logging_func(f"Samples per epoch: {samples_per_epoch}")
        self.logging_func(f"Num. URLs: {len(url_list)}")
        self.logging_func(f"From Cloud: {from_cloud}")

        self.VOCAB = vocabulary
        self._pad_token_id = self.VOCAB["PAD"]
        self._eos_token_id = self.VOCAB["EOS"]

        # TODO: Take list of already consumed lists and remove them from the
        # url list, to continue training from the last checkpoint properly

        if from_cloud:
            # In order to make sure data are shuffled and sharded in the
            # distributed environment, `shuffle`  and `sharding_filter`
            # are required. For detail, please check our tutorial in:
            # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
            dp_s3_urls = IterableWrapper(url_list) # .list_files_by_s3()
            sharded_s3_urls = dp_s3_urls.shuffle().sharding_filter()

            # opened_files = S3FileLoader(sharded_s3_urls)
            opened_files = FSSpecFileOpener(sharded_s3_urls, mode="rb")

        else:
            files_names = IterableWrapper(url_list).shuffle().sharding_filter()
            opened_files = FileOpener(files_names, mode="rb")

        decoded_files = StreamReader(opened_files)
        decompressed_files = Mapper(decoded_files, self.decompression_func)

        self.prefetch_decompressed_files = prefetch_decompressed_files
        if self.prefetch_decompressed_files > 0:
            self.logging_func(f"Prefetching {self.prefetch_decompressed_files} decompressed files")
            # decompressed_files = Prefetcher(decompressed_files, self.prefetch_decompressed_files)
            decompressed_files = decompressed_files.prefetch(self.prefetch_decompressed_files)

        sequences_batched = Mapper(decompressed_files, self.fasta_parsing_func)
        self.prefetch_fasta_parsing = prefetch_fasta_parsing
        if self.prefetch_fasta_parsing > 0:
            self.logging_func(f"Prefetching {self.prefetch_fasta_parsing} parsed sequences")
            sequences_batched = sequences_batched.prefetch(self.prefetch_fasta_parsing)

        sequences_unbatched = sequences_batched.unbatch()
        self.prefetch_sequences = prefetch_sequences
        if self.prefetch_sequences > 0:
            self.logging_func(f"Prefetching {self.prefetch_sequences} unbatched sequences")
            sequences_unbatched = sequences_unbatched.prefetch(self.prefetch_sequences)

        # sequences_crop = Mapper(sequences_unbatched, self.crop_maxlen)
        # sequences_tokenized = Mapper(sequences_crop, self.tokenize_and_pad)

        # if self.prefetch_sequences > 0:
        #     self.logging_func(f"Prefetching {self.prefetch_sequences} sequences")
        #     # sequences_tokenized = Prefetcher(sequences_tokenized, self.prefetch_sequences)
        #     sequences_tokenized = sequences_tokenized.prefetch(self.prefetch_sequences)

        if from_cloud:
            self.iterable_dataset = iter(sequences_unbatched)
        else:
            self.iterable_dataset = self.cyclic_iter(sequences_unbatched)
        self.logging_func(f"Sample: {next(self.iterable_dataset)}")

        self.logging_func(f"Pipeline warmup:")
        warmup_sample_size = 1024
        for _ in range(warmup_sample_size):
            _ = next(self.iterable_dataset)

        self.logging_func("=====================================")

    def decompression_func(self, input_data):
        path, data = input_data
        # if self.debug:
        #     self.logging_func(f"[{self.__class__.__name__}] Decompressing {path}")

        decompressed_data = zstd.decompress(data)
        # if self.debug:
        #     num_mb_compressed = len(data) / 1024 / 1024
        #     num_mb_decompressed = len(decompressed_data) / 1024 / 1024
        #     self.logging_func(f"[{self.__class__.__name__}] Decompressed {num_mb_compressed:.2f} MB to {num_mb_decompressed:.2f} MB for {path}")
        
        return path, decompressed_data

    def fasta_parsing_func(self, input_data):
        path, data = input_data

        # if self.debug:
        #     self.logging_func(f"[{self.__class__.__name__}] Parsing {path}")

        sequences = []
        current_sequence = ""
        decoded_lines = data.decode() # .split("\n")
        # if self.debug:
        #     self.logging_func(f"[{self.__class__.__name__}] Found {len(decoded_lines)} lines in {path}")

        sequences = [(path, str(s.seq)) for s in SeqIO.parse(StringIO(decoded_lines), "fasta")]

        # if self.debug:
        #     self.logging_func(f"[{self.__class__.__name__}] Found {len(sequences)} sequences in {path}")
        #     self.logging_func(f"[{self.__class__.__name__}] First sequence: {sequences[0]}")

        return sequences

    def crop_maxlen(self, input_sequence: str):
        # path, input_sequence = input_data
        maxlen_without_special_tokens = self.maxlen - 2
        if len(input_sequence) <= maxlen_without_special_tokens:
            return input_sequence
        else:
            # Crop the sequence to the maximum length
            # Get random starting point
            start = random.randint(0, len(input_sequence) - maxlen_without_special_tokens)
            return input_sequence[start:start + maxlen_without_special_tokens]

    def tokenize_and_pad(self, input_sequence: str):
        # path, input_sequence = input_data
        maxlen = self.maxlen

        # Tokenize the sequence
        tokenized_sequence = [0] # start with BOS token
        tokenized_sequence.extend([self.VOCAB.get(base, 3) for base in input_sequence]) # 3 is the UNK token
        tokenized_sequence.append(1) # end with EOS token
        tokenized_sequence = np.array(tokenized_sequence, dtype=np.int32)

        # Pad the sequence

        if len(tokenized_sequence) < maxlen:
            # 2 is the PAD token
            tokenized_sequence = np.pad(tokenized_sequence,
                                        (0, maxlen - len(tokenized_sequence)),
                                        mode="constant",
                                        constant_values=2)

        # if len(tokenized_sequence) < maxlen:
        #     tokenized_sequence.extend([2] * (maxlen - len(tokenized_sequence))) # 2 is the PAD token

        return tokenized_sequence
        
    # def __len__(self):
    #     return self.samples_per_epoch

    def generate(self):
        while True:
            # if self.debug:
            #     print(f"[{self.__class__.__name__}] Getting item {idx}")

            try:
                source_path, text_raw = next(self.iterable_dataset)
            except StopIteration:  
                self.logger.warning(f"Reached end of dataset, restarting from the beginning")
            text_cropped = self.crop_maxlen(text_raw)
            text_tokenized = self.tokenize_and_pad(text_cropped)

            yield {"input_ids": text_tokenized}
    
    # def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
    def __iter__(self) -> dict[str, np.ndarray]:

        """Abstract method implementation

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        return cyclic_iter(self.generate())



