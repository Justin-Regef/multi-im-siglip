import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union


# Helper function (assuming you have this or a similar utility)
def l2_normalize_tensor(
    x: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """L2 normalizes a tensor along a given dimension."""
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    return x / (norm + eps)


class SigLipLossWithAmbiguityForOpenClip(nn.Module):
    def __init__(
        self,
        # cache_labels: bool = False, # Not directly used by this ambiguity logic
        rank: int = 0,  # For context, not used in this simplified distributed handling
        world_size: int = 1,  # For context
        # dist_impl: Optional[str] = None, # Distributed strategy handled by data prep
        pick_best_candidate: bool = True,  # New parameter for your ambiguity logic
    ):
        super().__init__()
        # self.cache_labels = cache_labels # Caching labels like in original OpenCLIP might be complex with dynamic key
        self.rank = rank
        self.world_size = world_size
        self.pick_best_candidate = pick_best_candidate
        # self.prev_num_logits = 0 # Not used in this version
        # self.labels = {} # Not used in this version

    def forward(
        self,
        image_features: torch.Tensor,  # s x dim_img (s = number of images in batch)
        text_features: torch.Tensor,  # n x dim_txt (n = number of unique texts in batch)
        key: torch.Tensor,  # s (long tensor, mapping each image_features[i] to a text_features index)
        logit_scale: torch.Tensor,  # Scalar learned temperature (already exp'd)
        logit_bias: Optional[torch.Tensor] = None,  # Scalar learned bias
        output_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            image_features: Embeddings of images, shape (s, dim).
            text_features: Embeddings of texts, shape (n, dim).
            key: Tensor of shape (s,), where key[i] is the index of the text
                 embedding in text_features that image_features[i] corresponds to.
                 Assumes 0 <= key[i] < n.
            logit_scale: The temperature parameter (already exponentiated).
            logit_bias: The bias parameter.
            output_dict: Whether to return a dictionary with the loss.
        """
        s, dim_img = image_features.shape
        n, dim_txt = text_features.shape
        device = image_features.device
        dtype = image_features.dtype

        if dim_img != dim_txt:
            raise ValueError(
                f"Image and text embedding dimensions must match ({dim_img} vs {dim_txt})."
            )
        dim = dim_img

        if key.max() >= n or key.min() < 0:
            # Basic check, more thorough checks might be needed depending on data
            if n == 0 and s == 0 and key.numel() == 0:  # handle empty batch case
                pass
            elif n == 0 and s > 0:
                raise ValueError(
                    f"Key references text indices but there are no text features (n=0, s={s})."
                )
            elif key.numel() > 0:  # only check if key is not empty
                raise ValueError(
                    f"Key contains indices out of bounds for text_features. Max key index: {key.max()}, num_texts (n): {n}"
                )

        # Normalize features - OpenCLIP models usually output normalized features,
        # but explicit normalization here is safer if unsure.
        zimg = l2_normalize_tensor(image_features, dim=-1)
        ztxt = l2_normalize_tensor(text_features, dim=-1)

        # Calculate all s x n pairwise logits
        # These are logits between ALL images and ALL unique texts in the batch
        all_pairs_logits = torch.matmul(zimg, ztxt.T) * logit_scale
        if logit_bias is not None:
            all_pairs_logits = all_pairs_logits + logit_bias

        # For each of the 's' images, get the logit corresponding to its assigned text caption (via 'key')
        # true_positive_logits will have shape (s,)
        true_positive_logits = all_pairs_logits[torch.arange(s, device=device), key]

        # Calculate potential losses for these true positive pairings.
        # This helps in the selection step if pick_best_candidate is True.
        # F.softplus(-x) is equivalent to -F.logsigmoid(x) for positive examples.
        potential_positive_losses = F.softplus(-true_positive_logits)  # Shape (s,)

        # --- Ambiguity Resolution: Select one image per text prompt ---
        # zimg_selected will hold the chosen image embedding for each of the 'n' unique text prompts
        zimg_selected = torch.zeros((n, dim), dtype=zimg.dtype, device=device)
        # selected_img_global_indices stores which of the 's' images was chosen for each 'n' text
        selected_img_global_indices = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )
        # valid_text_mask indicates which text prompts had at least one associated image
        valid_text_mask = torch.zeros(n, dtype=torch.bool, device=device)

        if s > 0 and n > 0:  # Only proceed if there are images and texts
            for j in range(n):  # Iterate over each unique text prompt
                # Find all images 'i' (from 0 to s-1) that are associated with the current text 'j'
                candidate_img_indices = torch.nonzero(key == j, as_tuple=False).squeeze(
                    -1
                )

                if candidate_img_indices.numel() == 0:
                    # This text prompt 'j' has no associated images in the current batch.
                    # zimg_selected[j] will remain a zero vector.
                    # selected_img_global_indices[j] will remain -1.
                    continue

                valid_text_mask[j] = True
                # Get the pre-calculated potential losses for these candidate images
                losses_for_candidates = potential_positive_losses[candidate_img_indices]

                if self.pick_best_candidate:
                    # Select the image that has the MINIMUM loss (i.e., highest logit/agreement)
                    # with the current text prompt 'j'.
                    best_candidate_local_idx = torch.argmin(losses_for_candidates)
                else:
                    # Select the image that has the MAXIMUM loss (i.e., lowest logit/agreement)
                    # to ensure even challenging pairs are attended to.
                    best_candidate_local_idx = torch.argmax(losses_for_candidates)

                # Get the global index (from 0 to s-1) of the chosen image
                best_global_img_idx = candidate_img_indices[best_candidate_local_idx]

                selected_img_global_indices[j] = best_global_img_idx
                zimg_selected[j] = zimg[best_global_img_idx]

        # If no texts had any valid images, the loss is 0.
        # This can happen if n > 0 but all texts have no associated images in the 'key'.
        num_valid_texts = valid_text_mask.sum()
        if num_valid_texts == 0:
            final_loss = torch.tensor(
                0.0, device=device, dtype=dtype, requires_grad=True
            )
            if output_dict:
                return {
                    "contrastive_loss": final_loss,
                    "selected_indices": selected_img_global_indices,  # for debugging
                    "num_valid_texts": num_valid_texts,
                }
            return final_loss

        # --- Final Loss Calculation (SigLIP-style) ---
        # We now have 'n' selected image embeddings (in zimg_selected), one for each unique text.
        # Some entries in zimg_selected might be zero if a text had no images.
        # We compute an n x n logit matrix: (selected_images) vs (all_texts)
        # We only consider texts that had at least one image associated with them for the loss.

        # Filter down to valid texts and their selected images
        active_ztxt = ztxt[valid_text_mask]  # num_valid_texts x dim
        active_zimg_selected = zimg_selected[valid_text_mask]  # num_valid_texts x dim

        # Final n_valid x n_valid logits matrix
        final_logits_matrix = (
            torch.matmul(active_zimg_selected, active_ztxt.T) * logit_scale
        )
        if logit_bias is not None:
            # Apply bias carefully if it's shared or per-output
            # Assuming logit_bias is a scalar or compatible
            final_logits_matrix = final_logits_matrix + logit_bias

        # Create ground truth labels for this n_valid x n_valid matrix.
        # Diagonals are positive (label +1), off-diagonals are negative (label -1).
        # Equivalent to (2 * I - 1)
        final_labels = 2 * torch.eye(
            num_valid_texts, device=device, dtype=dtype
        ) - torch.ones((num_valid_texts, num_valid_texts), device=device, dtype=dtype)

        # Sigmoid loss: L = -log(sigmoid(label * logit)) = softplus(-label * logit)
        loss_terms = F.softplus(-(final_labels * final_logits_matrix))

        # Sum losses and normalize by the number of valid text samples
        final_loss = torch.sum(loss_terms) / num_valid_texts

        if output_dict:
            # Return more info if needed for analysis
            return {
                "contrastive_loss": final_loss,
                "selected_indices": selected_img_global_indices,  # Full, includes -1 for non-matched
                "final_logits": final_logits_matrix,  # Logits for valid texts
                "labels": final_labels,  # Labels for valid texts
                "num_valid_texts": num_valid_texts,
            }
        return final_loss
