import logging
import pandas as pd
from PIL import Image
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataclasses import dataclass


# The training loop will look something like this:
# for batch_idx, (images, texts_tokenized, key) in enumerate(dataloader):
#     images = images.to(device)
#     texts_tokenized = texts_tokenized.to(device)
#     key = key.to(device)
#
#     # Pass through OpenCLIP model
#     image_embeddings, text_embeddings, logit_scale = model(images, texts_tokenized)
#     # logit_scale is learned by the model.
#     # text_embeddings here will be from the unique tokenized texts (shape: n x embed_dim)
#     # image_embeddings here will be from all images (shape: s x embed_dim)
#
#     # Calculate loss
#     # The `text_embeddings` from the model correspond to `batch_unique_texts_tokenized`.
#     # The `image_embeddings` from the model correspond to `batch_images`.
#     # The `key` correctly maps these.
#     loss = loss_fn(
#         image_features=image_embeddings,
#         text_features=text_embeddings,  # These are embeddings of the unique texts
#         key=key,
#         logit_scale=logit_scale,
#         logit_bias=getattr(model, "logit_bias", None),  # If your model has a logit_bias
#     )
#     # ... rest of training loop ...


# Assuming open_clip.shared_epoch.SharedEpoch is available or a similar mechanism
# For simplicity, if not available, we can remove SharedEpoch parts or mock it.
try:
    from open_clip.shared_epoch import SharedEpoch
except ImportError:

    class SharedEpoch:  # Dummy class if not available
        def __init__(self, epoch: int = 0):
            self.shared_epoch = epoch

        def set_value(self, epoch):
            self.shared_epoch = epoch

        def get_value(self):
            return self.shared_epoch

    logging.warning(
        "open_clip.shared_epoch.SharedEpoch not found, using a dummy implementation."
    )


class CsvDatasetForAmbiguity(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        """
        Dataset that loads images and raw caption strings.
        Tokenization is deferred to the collate_fn.
        """
        logging.debug(f"Loading csv data from {input_filename}.")
        try:
            df = pd.read_csv(input_filename, sep=sep)
        except FileNotFoundError:
            logging.error(f"CSV file not found: {input_filename}")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV file {input_filename}: {e}")
            raise

        if img_key not in df.columns:
            raise ValueError(
                f"Image key '{img_key}' not found in CSV columns: {df.columns.tolist()}"
            )
        if caption_key not in df.columns:
            raise ValueError(
                f"Caption key '{caption_key}' not found in CSV columns: {df.columns.tolist()}"
            )

        self.images_paths = df[img_key].tolist()
        self.raw_captions = df[caption_key].tolist()  # Store raw captions
        self.transforms = transforms
        logging.info(
            f"Done loading data from {input_filename}. Found {len(self.raw_captions)} samples."
        )

    def __len__(self):
        return len(self.raw_captions)

    def __getitem__(self, idx):
        try:
            image_path = str(self.images_paths[idx])
            # Ensure image path is not NaN or empty
            if pd.isna(image_path) or not image_path.strip():
                raise ValueError(
                    f"Invalid or empty image path at index {idx}: '{image_path}'"
                )

            image = Image.open(image_path)
            image_tensor = self.transforms(image)
        except FileNotFoundError:
            logging.warning(
                f"Image file not found: {self.images_paths[idx]}. Skipping sample or returning placeholder."
            )
            # Option 1: Raise error to be caught by collate_fn or dataloader
            # raise
            # Option 2: Return a placeholder (less ideal for training)
            # For now, let's assume paths are valid or an error is preferred.
            # If you want to handle this gracefully, you might need to skip indices in collate_fn.
            # However, standard datasets usually expect __getitem__ to succeed or fail clearly.
            raise RuntimeError(
                f"Failed to load image: {self.images_paths[idx]}"
            ) from FileNotFoundError
        except Exception as e:
            raise RuntimeError(
                f"Error processing image {self.images_paths[idx]} at index {idx}: {e}"
            )

        raw_caption = str(self.raw_captions[idx])
        # Ensure caption is not NaN or empty, though an empty string might be a valid caption for some use cases
        if pd.isna(raw_caption):
            logging.warning(
                f"NaN caption found at index {idx} for image {self.images_paths[idx]}. Replacing with empty string."
            )
            raw_caption = ""  # Or handle as an error / skip

        return image_tensor, raw_caption


def collate_fn_for_ambiguity(batch, tokenizer):
    """
    Collate function to prepare batches for SigLipLossWithAmbiguity.
    batch: A list of (image_tensor, raw_caption_string) tuples.
    tokenizer: The tokenizer function (e.g., from open_clip.tokenizer).
    """
    images, raw_captions = zip(*batch)

    # Stack image tensors: (s, C, H, W) where s is batch_size
    images_tensor = torch.stack(images)

    # Identify unique raw captions and create mapping
    unique_raw_captions = sorted(list(set(raw_captions)))
    caption_to_idx = {caption: i for i, caption in enumerate(unique_raw_captions)}

    # Create key tensor: (s,)
    # key[i] is the index of the unique caption for images_tensor[i]
    key_tensor = torch.tensor(
        [caption_to_idx[caption] for caption in raw_captions], dtype=torch.long
    )

    # Tokenize unique captions: (n, seq_len) where n is num_unique_captions
    # The tokenizer typically returns a tensor of shape (num_texts, context_length)
    unique_texts_tokenized = tokenizer(unique_raw_captions)

    return images_tensor, unique_texts_tokenized, key_tensor


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None  # Keep for compatibility

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)  # type: ignore
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset_for_ambiguity(
    args, preprocess_fn, is_train, epoch=0, tokenizer=None
):
    input_filename = args.train_data if is_train else args.val_data
    if not input_filename:
        raise ValueError(
            "Input filename (args.train_data or args.val_data) is not provided."
        )
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for CsvDatasetForAmbiguity.")

    dataset = CsvDatasetForAmbiguity(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        # Tokenizer is not passed to dataset, but used in collate_fn
    )
    num_samples = len(dataset)

    sampler = None
    if args.distributed and is_train:
        sampler = DistributedSampler(dataset)

    shuffle = is_train and sampler is None

    # Use functools.partial to pass the tokenizer to the collate_fn
    custom_collate_fn = partial(collate_fn_for_ambiguity, tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=custom_collate_fn,  # Use the custom collate function
    )
    # TODO: these attributes are not part of DataLoader by default.
    # Consider if they are truly needed or how to best provide this info.
    # For now, adding them as attributes as in the original snippet.
    try:
        dataloader.num_samples = num_samples  # type: ignore
        dataloader.num_batches = len(dataloader)  # type: ignore
    except AttributeError:  # pragma: no cover
        # some dataloaders (e.g. IterableDataset) may not support this.
        # We only use this for logging anyway so it's not critical.
        pass

    shared_epoch = None
    if (
        hasattr(args, "use_shared_epoch") and args.use_shared_epoch
    ):  # Assuming such an arg might exist
        # If using webdataset, a SharedEpoch can be used to synchronize epoch numbers
        # across worker processes for curriculum learning. Not strictly necessary for CSV.
        shared_epoch = SharedEpoch(epoch=epoch)

    return DataInfo(dataloader, sampler, shared_epoch=shared_epoch)  # type: ignore
