import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def visualize_captions(
    image_paths: list,
    rnn_captions: list,
    lstm_captions: list,
    references: list,
    scores_rnn: list,
    scores_lstm: list,
    selected_indices: list,
    save_path: str = None,
    figsize_per_row: tuple = (14, 4),
):
    n = len(selected_indices)
    fig_h = figsize_per_row[1] * n
    fig, axes = plt.subplots(n, 2, figsize=(figsize_per_row[0], fig_h),
                             gridspec_kw={"width_ratios": [1, 2]})

    if n == 1:
        axes = [axes]

    for row_ax, idx in zip(axes, selected_indices):
        img_ax, txt_ax = row_ax

        # Image
        try:
            img = mpimg.imread(image_paths[idx])
            img_ax.imshow(img)
        except Exception:
            img_ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
        img_ax.axis("off")
        img_ax.set_title(f"Image #{idx}", fontsize=9, pad=3)

        # Captions text
        gt = references[idx][0] if references[idx] else "—"
        caption_text = (
            f"RNN  (BLEU-4: {scores_rnn[idx]:.3f})\n"
            f"  {rnn_captions[idx]}\n\n"
            f"LSTM (BLEU-4: {scores_lstm[idx]:.3f})\n"
            f"  {lstm_captions[idx]}\n\n"
            f"Ground Truth\n"
            f"  {gt}"
        )
        txt_ax.text(
            0.02, 0.95, caption_text,
            transform=txt_ax.transAxes,
            fontsize=10, verticalalignment="top",
            wrap=True,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        txt_ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def select_qualitative_samples(
    per_image_scores_rnn: list,
    per_image_scores_lstm: list,
    n_low: int = 3,
    n_mid: int = 4,
    n_high: int = 3,
) -> list:
    avg_scores = [
        (i, (r + l) / 2)
        for i, (r, l) in enumerate(zip(per_image_scores_rnn, per_image_scores_lstm))
    ]
    avg_scores.sort(key=lambda x: x[1])

    n = len(avg_scores)
    low_idx = [x[0] for x in avg_scores[:n_low]]
    mid_start = n // 4
    mid_step = max(1, (n // 2) // n_mid)
    mid_idx = [avg_scores[mid_start + j * mid_step][0] for j in range(n_mid)
               if mid_start + j * mid_step < 3 * n // 4]
    high_idx = [x[0] for x in avg_scores[-n_high:]]

    return low_idx + mid_idx + high_idx
