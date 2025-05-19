import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --- Helper function ---
@torch.jit.script
def l2_normalize_tensor(x, dim: int = -1, eps: float = 1e-12):
    """L2 Normalizes a tensor along a given dimension."""
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


# --- Custom Loss Module (using the version robust to torch.nonzero issues) ---
class SigLipLossWithAmbiguity(nn.Module):
    def __init__(self, initial_log_t: float = 0.0, initial_bias: float = 0.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(initial_log_t, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

    def forward(
        self, img_emb_raw: torch.Tensor, txt_emb_raw: torch.Tensor, key: torch.Tensor
    ):
        s, dim_img = img_emb_raw.shape
        n, dim_txt = txt_emb_raw.shape
        device = img_emb_raw.device

        if dim_img != dim_txt:
            raise ValueError(
                f"Image and text embedding dimensions must match ({dim_img} vs {dim_txt})."
            )
        dim = dim_img

        t = torch.exp(self.log_t)
        zimg = l2_normalize_tensor(img_emb_raw, dim=-1)
        ztxt = l2_normalize_tensor(txt_emb_raw, dim=-1)

        all_pairs_logits = torch.matmul(zimg, ztxt.T) * t + self.bias
        true_positive_logits = all_pairs_logits[torch.arange(s, device=device), key]
        potential_positive_losses = F.softplus(-true_positive_logits)

        zimg_selected = torch.zeros((n, dim), dtype=zimg.dtype, device=device)
        selected_img_global_indices = torch.full(
            (n,), -1, dtype=torch.long, device=device
        )

        for j in range(n):
            # Using the variant confirmed to work by the user (no as_tuple=False)
            candidate_img_indices = torch.nonzero(key == j).squeeze(-1)

            if candidate_img_indices.numel() == 0:
                # This text sample has no associated images. zimg_selected[j] remains zero.
                # selected_img_global_indices[j] remains -1.
                continue

            losses_for_candidates = potential_positive_losses[candidate_img_indices]

            # Now we either find the best loss and propagate only that -> might lead to degenerate greedy only picking the same pairs
            # Or we pick the worst loss and propagate it -> to ensure everything is attended to envenly
            # In SAM they pick the best candidate
            pick_best = True
            if pick_best:
                best_candidate_local_idx = torch.argmin(losses_for_candidates)
            else:
                best_candidate_local_idx = torch.argmax(losses_for_candidates)
            best_global_img_idx = candidate_img_indices[best_candidate_local_idx]

            selected_img_global_indices[j] = best_global_img_idx
            zimg_selected[j] = zimg[best_global_img_idx]

        final_logits_matrix = (
            torch.matmul(zimg_selected, ztxt.T) * t + self.bias
        )  # Renamed for clarity
        final_labels = 2 * torch.eye(n, device=device) - torch.ones(
            (n, n), device=device
        )
        loss_terms = F.softplus(-(final_labels * final_logits_matrix))

        if n == 0:
            final_loss = torch.tensor(0.0, device=device, dtype=img_emb_raw.dtype)
        else:
            final_loss = torch.sum(loss_terms) / n

        # Return selected indices and other useful tensors for analysis if needed
        # For the sanity check, we'll need zimg, ztxt, t, bias, and selected_img_global_indices.
        # The loss function itself will primarily be used for its scalar loss during training.
        # For analysis, we can call forward and unpack these.
        return (
            final_loss,
            selected_img_global_indices,
            zimg,
            ztxt,
            all_pairs_logits,
            final_logits_matrix,
        )


# --- Dummy Models ---
class DummyImageModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class DummyTextModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# --- Training Script ---
def main_training_loop():
    # Hyperparameters
    emb_dim: int = 64
    n_texts: int = 4  # Keep small for easier inspection
    s_images: int = 8  # s_images >= n_texts
    dummy_input_dim: int = 124

    num_epochs: int = 10000  # Increased epochs to observe overfitting
    learning_rate: float = 1e-3  # Might need adjustment

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Fixed Dummy Batch Data (Moved outside the loop) ---
    # torch.manual_seed(42)  # For reproducible dummy data
    dummy_img_input = torch.randn(s_images, dummy_input_dim, device=device)
    dummy_txt_input = torch.randn(n_texts, dummy_input_dim, device=device)

    if s_images < n_texts:
        raise ValueError("s_images must be >= n_texts")
    key_parts = [torch.arange(n_texts, dtype=torch.long, device=device)]
    if s_images > n_texts:
        key_parts.append(
            torch.randint(
                0, n_texts, (s_images - n_texts,), dtype=torch.long, device=device
            )
        )
    key = torch.cat(key_parts, dim=0)
    key = key[torch.randperm(s_images)]
    print(f"Fixed Key: {key.tolist()}")
    # --- End Fixed Dummy Batch Data ---

    # Instantiate Models
    image_model = DummyImageModel(dummy_input_dim, emb_dim).to(device)
    text_model = DummyTextModel(dummy_input_dim, emb_dim).to(device)

    # Instantiate Loss Function
    loss_fn = SigLipLossWithAmbiguity(initial_log_t=0.0, initial_bias=0.0).to(device)
    # JIT scripting attempt (optional)
    try:
        # Note: Returning multiple tensors from a JIT scripted forward can sometimes be tricky
        # if not all paths return the same number/type of tensors, but here it's consistent.
        loss_fn_scripted = torch.jit.script(loss_fn)
        loss_fn_to_use = loss_fn_scripted
        print("Loss function JIT scripted successfully.")
    except Exception as e:
        print(f"Could not JIT script the loss function, using eager mode: {e}")
        loss_fn_to_use = loss_fn

    params_to_optimize = (
        list(image_model.parameters())
        + list(text_model.parameters())
        + list(loss_fn_to_use.parameters())
    )
    optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

    print(f"\nTraining for {num_epochs} epochs with fixed data...")
    image_model.train()
    text_model.train()
    loss_fn_to_use.train()  # Although log_t and bias are params, good practice

    for epoch in range(num_epochs):
        img_emb_raw = image_model(dummy_img_input)
        txt_emb_raw = text_model(dummy_txt_input)

        # Get loss and other info (selected_indices is primarily for post-training analysis here)
        loss, _, _, _, _, _ = loss_fn_to_use(img_emb_raw, txt_emb_raw, key)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % (num_epochs // 10) == 0 or epoch == 0:
            # Access parameters directly from the original module if JIT scripted one is used
            current_log_t = (
                loss_fn.log_t.item()
                if hasattr(loss_fn, "log_t")
                else loss_fn_to_use.log_t.item()
            )
            current_bias = (
                loss_fn.bias.item()
                if hasattr(loss_fn, "bias")
                else loss_fn_to_use.bias.item()
            )
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
                f"Learned log_t: {current_log_t:.4f}, Learned bias: {current_bias:.4f}"
            )

    final_log_t = (
        loss_fn.log_t.item()
        if hasattr(loss_fn, "log_t")
        else loss_fn_to_use.log_t.item()
    )
    final_bias = (
        loss_fn.bias.item() if hasattr(loss_fn, "bias") else loss_fn_to_use.bias.item()
    )
    print("\nTraining finished.")
    print(
        f"Final learned log_t: {final_log_t:.4f} (effective temperature t={torch.exp(torch.tensor(final_log_t)).item():.4f})"
    )
    print(f"Final learned bias: {final_bias:.4f}")

    # --- Sanity Check Post-Training ---
    print("\n--- Sanity Check on Trained Model ---")
    image_model.eval()
    text_model.eval()
    loss_fn_to_use.eval()  # Set loss module to eval mode

    with torch.no_grad():  # No need to compute gradients for sanity check
        img_emb_raw_final = image_model(dummy_img_input)
        txt_emb_raw_final = text_model(dummy_txt_input)

        # Get all outputs from the loss function for detailed analysis
        (
            final_loss_check,
            selected_indices,
            zimg_final,
            ztxt_final,
            all_pairs_logits_final,
            final_logits_matrix_check,
        ) = loss_fn_to_use(img_emb_raw_final, txt_emb_raw_final, key)

        print(f"Loss on fixed data after training: {final_loss_check.item():.4f}")

        effective_t = torch.exp(torch.tensor(final_log_t)).item()

        print("\nSelected Image for Each Text (and their logits):")
        for j in range(n_texts):
            selected_img_idx_for_text_j = selected_indices[j].item()

            print(f"\nText Sample {j}:")
            if selected_img_idx_for_text_j == -1:
                print(
                    "  No image was selected (this shouldn't happen if s_images >= n_texts and key is valid)."
                )
                continue

            print(
                f"  Selected Image Index (from s_images): {selected_img_idx_for_text_j}"
            )

            # Verify the key constraint
            original_text_for_selected_image = key[selected_img_idx_for_text_j].item()
            print(
                f"  Original text this image was keyed to: {original_text_for_selected_image} "
                f"{'(Matches current text sample - GOOD)' if original_text_for_selected_image == j else '(MISMATCH - selection logic error or key issue)'}"
            )

            # Logit for the selected_image[j] with text[j]
            # This is directly available from final_logits_matrix_check[j, j]
            logit_selected_pair = final_logits_matrix_check[j, j].item()
            print(
                f"  Logit(selected_img_for_text_{j}, text_{j}): {logit_selected_pair:.4f} (Should be high and positive)"
            )

            # For deeper insight, check logits of this selected image with other texts
            # And check logits of this text with other selected images (from final_logits_matrix_check)
            print(
                f"  Logits for Text {j} with ALL selected images (row {j} of final_logits_matrix):"
            )
            for k_text_idx in range(n_texts):
                log_val = final_logits_matrix_check[
                    j, k_text_idx
                ].item()  # Logit(selected_img_for_text_j, text_k)
                is_positive_pair = "(Positive Pair)" if k_text_idx == j else ""
                print(
                    f"    Logit(selected_img_for_text_{j}, text_{k_text_idx}): {log_val:.4f} {is_positive_pair}"
                )

            # You can also examine the 'all_pairs_logits_final' which shows raw similarities before selection
            print(
                f"  All candidate images for Text {j} and their `true_positive_logits` (scaled sim with Text {j}):"
            )
            candidate_img_indices_for_j = torch.nonzero(key == j).squeeze(-1)
            if candidate_img_indices_for_j.numel() > 0:
                for img_idx_candidate in candidate_img_indices_for_j:
                    true_logit_val = all_pairs_logits_final[
                        img_idx_candidate.item(), j
                    ].item()
                    is_selected = (
                        "<- SELECTED"
                        if img_idx_candidate.item() == selected_img_idx_for_text_j
                        else ""
                    )
                    print(
                        f"    Img {img_idx_candidate.item()} (keyed to Text {key[img_idx_candidate.item()].item()}): true_positive_logit with Text {j} = {true_logit_val:.4f} {is_selected}"
                    )
            else:
                print(f"    No candidate images found for Text {j} in the key.")

        # Print softmax of dot products between all image and text embeddings for each text embedding
        print(
            "\nSoftmax of dot products between all image embeddings and each text embedding:"
        )
        dot_products = torch.matmul(
            img_emb_raw_final, txt_emb_raw_final.T
        )  # [n_images, n_texts]
        softmax_per_text = torch.softmax(
            dot_products, dim=0
        )  # Softmax over images for each text

        for j in range(n_texts):
            print(f"\nText {j}:")
            for i in range(dot_products.shape[0]):
                print(
                    f"  Image {i}: dot = {dot_products[i, j].item():.4f}, softmax = {softmax_per_text[i, j].item():.4f} {'-> keyed' if key[i] == j else ''}"
                )


if __name__ == "__main__":
    main_training_loop()
