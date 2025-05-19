import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings  # For the missing images warning


@torch.jit.script
def l2_normalize(x, dim: int = -1, eps: float = 1e-12):
    """L2 Normalizes a tensor along a given dimension."""
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class SigLipLossWithAmbiguityResolution(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    @torch.jit.script
    def compute_loss(
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        key: torch.Tensor,
        t_prime: torch.Tensor,
        b: torch.Tensor,
    ):
        # Hyperparameters / Constants (derived from inputs)
        s, dim = img_emb.shape
        n = txt_emb.shape[0]
        device = img_emb.device  # Get device from input tensors

        # --- I. Preprocessing ---
        # 1. t = exp(t_prime)
        t = torch.exp(t_prime)

        # 2. zimg = l2_normalize(img_emb)
        zimg = l2_normalize(img_emb, dim=-1)  # Shape [s, dim]

        # 3. ztxt = l2_normalize(txt_emb)
        ztxt = l2_normalize(txt_emb, dim=-1)  # Shape [n, dim]

        # --- II. Ambiguity Resolution: Select the Best Image for Each Text ---

        # 4. all_pairs_logits = dot(zimg, ztxt.T) * t + b
        all_pairs_logits = torch.matmul(zimg, ztxt.T) * t + b  # Shape [s, n]

        # 5. true_positive_logits = all_pairs_logits[arange(s), key]
        true_positive_logits = all_pairs_logits[
            torch.arange(s, device=device), key
        ]  # Shape [s]

        # 6. potential_positive_losses = -log_sigmoid(true_positive_logits)
        potential_positive_losses = F.softplus(-true_positive_logits)  # Shape [s]

        # 7. Initialize arrays
        zimg_selected = torch.zeros(
            (n, dim), dtype=zimg.dtype, device=device
        )  # Shape [n, dim]
        # Initialize tensor to store the global indices of selected images
        selected_img_global_indices = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )  # Shape [n]

        # 8. For each text sample, find the image associated with it
        for j in range(n):  # Iterate through each unique text sample
            # 10. candidate_img_indices = find_indices_where(key == j)
            candidate_img_indices = torch.nonzero(key == j).squeeze(-1)

            # 11. If candidate_img_indices is empty:
            if candidate_img_indices.numel() == 0:
                # selected_img_global_indices[j] remains -1
                # zimg_selected[j] remains zero vector
                continue  # Move to the next text sample

            # 18. losses_for_candidates = potential_positive_losses[candidate_img_indices]
            losses_for_candidates = potential_positive_losses[candidate_img_indices]

            # 20. best_candidate_local_idx = argmin(losses_for_candidates)
            best_candidate_local_idx = torch.argmin(losses_for_candidates)

            # 21. Get the global index (from 0 to s-1) of this best image.
            best_global_img_idx = candidate_img_indices[best_candidate_local_idx]

            # Store the selected global image index
            selected_img_global_indices[j] = best_global_img_idx

            # 23. Assign the embedding of this best image to represent text 'j'.
            zimg_selected[j] = zimg[best_global_img_idx]

        # --- III. Standard SigLip Loss Calculation with Selected Images ---

        # 26. final_logits = dot(zimg_selected, ztxt.T) * t + b
        final_logits = torch.matmul(zimg_selected, ztxt.T) * t + b  # Shape [n, n]

        # 27. Create SigLip labels
        final_labels = 2 * torch.eye(n, device=device) - torch.ones(
            (n, n), device=device
        )  # Shape [n, n]

        # 29. Calculate the loss for each pair.
        loss_terms = F.softplus(
            -(final_labels * final_logits)
        )  # Element-wise, shape [n, n]

        # 31. The final loss
        if n == 0:
            final_loss = torch.tensor(0.0, device=device, dtype=final_logits.dtype)
        else:
            final_loss = torch.sum(loss_terms) / n  # Scalar

        return (
            final_loss,
            zimg_selected,
            ztxt,
            final_logits,
            selected_img_global_indices,
        )


def main():
    # Hyperparameters / Constants:
    dim: int = 256
    n: int = 10
    s: int = 16

    # torch.manual_seed(42)

    img_emb = torch.randn(s, dim)
    txt_emb = torch.randn(n, dim)

    if s < n:
        raise ValueError(
            "s must be greater than or equal to n to ensure all texts can be mapped."
        )

    key_parts = [torch.arange(n, dtype=torch.long)]
    if s > n:
        key_parts.append(torch.randint(0, n, (s - n,), dtype=torch.long))
    key = torch.cat(key_parts, dim=0)
    key = key[torch.randperm(s)]

    # force the algo to have a really good img vs a bad one for the same text to see if it will select it
    key[0] = 0
    key[1] = 0
    img_emb[0] = txt_emb[0] + torch.randn_like(txt_emb[0]) * 0.001

    t_prime = torch.randn(())
    b = torch.randn(())

    print("--- Inputs ---")
    print(f"Image embeddings shape (s, dim): [{s}, {dim}]")
    print(f"Text embeddings shape (n, dim): [{n}, {dim}]")
    print(f"Key shape (s,): [{s},]")
    print(f"Key (first 20 elements): {key[:20]}")
    print(f"Unique values in key: {torch.unique(key)}")
    print(f"Initial t_prime: {t_prime.item():.4f}")
    print(f"Initial b: {b.item():.4f}")
    print("-" * 20)

    algorithm_fn = SigLipLossWithAmbiguityResolution.compute_loss

    # Run the algorithm and get selected indices
    final_loss, zimg_selected, ztxt_normalized, final_logits, selected_indices = (
        algorithm_fn(img_emb, txt_emb, key, t_prime, b)
    )

    print("\n--- Outputs ---")
    print(f"zimg_selected shape: {zimg_selected.shape}")
    # print(f"ztxt_normalized shape: {ztxt_normalized.shape}") # ztxt is returned as is from input
    print(f"Final Logits (shape {final_logits.shape}):")
    # print(final_logits)
    print(f"\nSelected Image Indices (shape {selected_indices.shape}):")
    print(selected_indices)
    print(f"\nFinal Loss: {final_loss.item():.4f}")

    # Verification:
    # For each text j, if selected_indices[j] is not -1,
    # then zimg_selected[j] should be equal to l2_normalize(img_emb)[selected_indices[j]]
    normalized_img_emb_for_check = l2_normalize(img_emb, dim=-1)
    for j_idx in range(n):
        selected_idx = selected_indices[j_idx].item()
        if selected_idx != -1:
            # Compare the selected embedding with the original normalized embedding at the selected index
            is_matching = torch.allclose(
                zimg_selected[j_idx],
                normalized_img_emb_for_check[selected_idx],
                atol=1e-6,
            )
            if not is_matching:
                warnings.warn(
                    f"Verification failed for text {j_idx}: "
                    f"zimg_selected[{j_idx}] does not match the source embedding at index {selected_idx}"
                )
            # else:
            #     print(f"Verification successful for text {j_idx} (selected image index {selected_idx})")
        else:
            # This text sample had no associated images or some other issue
            is_zero_vector = torch.all(zimg_selected[j_idx] == 0).item()
            print(
                f"Text sample {j_idx} had no image selected (index: {selected_idx}). "
                f"Selected embedding is zero: {is_zero_vector}"
            )


if __name__ == "__main__":
    main()
