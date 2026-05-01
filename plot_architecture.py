"""AO-FRL system architecture diagram (paper Figure).

Single static matplotlib figure showing:
  - Data preparation (CIFAR-100 split into decoder pool / federated pool)
  - 20 Client Agents (encoder + DP pipeline)
  - A2A protocol bus (5 message types)
  - Server Agent (replay buffer + head + 5 hooks)
  - Evaluator Agent (test set, early stopping)
  - Privacy evaluation pipeline (decoder inversion + PSNR)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# ---- Color palette ----
COL_CLIENT = "#CFE8FF"       # light blue
COL_SERVER = "#FFE6CC"       # light orange
COL_EVAL = "#D5F0CF"         # light green
COL_DP = "#FFD9D9"           # light red (DP / privacy)
COL_BUFFER = "#FFFACC"       # light yellow (replay buffer)
COL_DATA = "#EEEEEE"         # gray (data / shared)
COL_BUS_LIGHT = "#E8E8E8"

ARROW_FWD = "#A22"           # client -> server
ARROW_REV = "#26A"           # server -> client (head/budget)
ARROW_EVAL = "#080"          # server -> evaluator
ARROW_DATA = "#888"          # data flow


def rbox(ax, xy, w, h, text, fc, fontsize=9, fontweight="normal",
         text_color="black", ec="black", lw=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                          fc=fc, ec=ec, linewidth=lw)
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color)


def arrow(ax, src, dst, color="black", style="->", lw=1.3, ls="-"):
    a = FancyArrowPatch(src, dst, arrowstyle=style,
                         mutation_scale=14, color=color,
                         linewidth=lw, linestyle=ls,
                         shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(17, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ===== Title =====
    ax.text(10, 13.5,
            "AO-FRL: Agent-Orchestrated Federated Representation Learning",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(10, 13.05,
            "20 client agents  ·  1 server agent  ·  1 evaluator agent  ·  A2A protocol bus",
            ha="center", fontsize=10, style="italic", color="#444")

    # ===== TOP STRIP: Data preparation =====
    rbox(ax, (0.4, 11.6), 2.4, 0.85,
         "CIFAR-100\n(50K train + 10K test)", COL_DATA, fontsize=8.5)
    arrow(ax, (2.8, 12.0), (3.5, 12.0))
    rbox(ax, (3.5, 11.6), 2.4, 0.85,
         "split_decoder_pool\n(seed=42)", COL_DATA, fontsize=8.5)
    # Branch
    arrow(ax, (5.9, 12.2), (7.0, 12.55))
    arrow(ax, (5.9, 11.85), (7.0, 11.55))
    rbox(ax, (7.0, 12.25), 2.6, 0.7,
         "Decoder Pool 5K  (server-only)", COL_DP, fontsize=8.5)
    rbox(ax, (7.0, 11.2), 2.6, 0.7,
         "Federated Pool 45K  (clients)", COL_CLIENT, fontsize=8.5)
    arrow(ax, (9.6, 11.55), (10.7, 11.55))
    rbox(ax, (10.7, 11.2), 2.6, 0.7,
         "Dirichlet  α=0.3\n→ 20 disjoint shards", COL_DATA, fontsize=8.5)
    arrow(ax, (9.6, 12.6), (11.5, 12.6), color=ARROW_DATA, ls=":")
    ax.text(10.55, 12.78, "decoder train", fontsize=7,
            ha="center", color="#900")
    rbox(ax, (11.5, 12.25), 2.4, 0.7,
         "Decoder g_ψ\n(for PSNR eval)", COL_DP, fontsize=8.5)
    rbox(ax, (14.7, 11.6), 2.6, 0.85,
         "Test set 10K\n→ Evaluator", COL_EVAL, fontsize=8.5)

    # Top -> client pool
    arrow(ax, (12.0, 11.2), (3.5, 9.5), color=ARROW_DATA, ls=":")
    ax.text(8, 10.4, "shards distributed to clients",
            fontsize=8, color=ARROW_DATA, ha="center", style="italic")
    # Top -> evaluator
    arrow(ax, (16.0, 11.6), (16.5, 9.55), color=ARROW_DATA, ls=":")

    # ===== MIDDLE LEFT: Client Agents =====
    cx, cy, cw, ch = 0.4, 4.7, 6.2, 4.8
    rbox(ax, (cx, cy), cw, ch, "", "white", lw=1.4)
    ax.text(cx + cw / 2, cy + ch - 0.3,
            "Client Agent  C_i   (×20, parallel)",
            ha="center", fontsize=11, fontweight="bold", color="#246")

    # Client pipeline (vertical)
    px = cx + 0.5
    py_top = cy + ch - 0.7
    step_w = 2.4
    step_h = 0.45

    rbox(ax, (px, py_top - 0.45), step_w, step_h,
         "Local data x_i  (PIL 32×32)", COL_CLIENT, fontsize=8.5)
    arrow(ax, (px + step_w / 2, py_top - 0.45),
              (px + step_w / 2, py_top - 0.85))
    rbox(ax, (px, py_top - 1.30), step_w, step_h,
         "Encoder f_φ  (frozen)", "#E8E8E8", fontsize=8.5)
    arrow(ax, (px + step_w / 2, py_top - 1.30),
              (px + step_w / 2, py_top - 1.70))
    rbox(ax, (px, py_top - 2.15), step_w, step_h,
         "z_i  (512-d, L2-norm)", COL_CLIENT, fontsize=8.5)
    arrow(ax, (px + step_w / 2, py_top - 2.15),
              (px + step_w / 2, py_top - 2.55))
    rbox(ax, (px, py_top - 3.00), step_w, step_h,
         "L2 clip  (||z|| ≤ C)", COL_DP, fontsize=8.5)
    arrow(ax, (px + step_w / 2, py_top - 3.00),
              (px + step_w / 2, py_top - 3.40))
    rbox(ax, (px, py_top - 3.85), step_w, step_h,
         "+  N(0, σ²I)   (DP noise)", COL_DP, fontsize=8.5)
    # Right: noisy z
    arrow(ax, (px + step_w, py_top - 3.62),
              (px + step_w + 0.7, py_top - 3.62))
    rbox(ax, (px + step_w + 0.7, py_top - 3.85), 2.0, step_h,
         "noisy z̃_i", COL_DP, fontsize=8.5, fontweight="bold")

    # Right side: per-class budget annotation
    ax.text(px + step_w + 1.7, py_top - 0.22,
            "Budget B_{i,c}\n(per-class\nupload cap)",
            ha="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#888", lw=0.6))
    arrow(ax, (px + step_w + 1.7, py_top - 0.85),
              (px + step_w + 1.7, py_top - 2.0),
              color="#444", ls="--")
    ax.text(px + step_w + 0.7, py_top - 1.4, "controls\nsampling",
            fontsize=7, color="#444", ha="left", style="italic")

    # ===== MIDDLE: A2A bus =====
    bus_x, bus_y, bus_w, bus_h = 7.05, 4.7, 1.0, 4.8
    ax.add_patch(Rectangle((bus_x, bus_y), bus_w, bus_h,
                           fc=COL_BUS_LIGHT, ec="black", linewidth=1.4))
    ax.text(bus_x + bus_w / 2, bus_y + bus_h - 0.25,
            "A2A\nProtocol", ha="center", fontsize=10,
            fontweight="bold", color="#222")
    bus_msgs = [
        ("histogram h_i", "↑"),
        ("z̃_i  (DP emb)", "↑"),
        ("per-class acc a_i", "↑"),
        ("head θ", "↓"),
        ("budget B_i", "↓"),
    ]
    for i, (lab, dirsym) in enumerate(bus_msgs):
        y = bus_y + bus_h - 1.0 - i * 0.75
        col = ARROW_FWD if dirsym == "↑" else ARROW_REV
        ax.text(bus_x + bus_w / 2, y, f"{dirsym} {lab}",
                ha="center", fontsize=7.5, color=col,
                fontweight="bold")

    # ===== MIDDLE RIGHT: Server Agent =====
    sx, sy, sw, sh = 8.55, 4.7, 7.4, 4.8
    rbox(ax, (sx, sy), sw, sh, "", "white", lw=1.4)
    ax.text(sx + sw / 2, sy + sh - 0.3, "Server Agent  S",
            ha="center", fontsize=11, fontweight="bold", color="#A60")

    # Replay buffer
    rbox(ax, (sx + 0.35, sy + 1.7), 2.7, 2.6,
         "Replay Buffer\n(N ≤ 500K)\n\nstores (z̃_j, y_j, r_j)\n\n"
         "weight\n  w_j = max(γ^(t-r_j), w_min)\n  γ=0.995, w_min=0.3",
         COL_BUFFER, fontsize=8)

    # Head
    rbox(ax, (sx + 3.25, sy + 2.65), 2.7, 1.65,
         "Head θ\nMLP 512 → 256 → 100\n\nWeightedRandomSampler\nLR_t = lr · 0.98^(t-1)",
         COL_SERVER, fontsize=8)

    # Hooks
    rbox(ax, (sx + 3.25, sy + 1.0), 2.7, 1.5,
         "Server Hooks\n• init_budget  (round 0)\n"
         "• feedback_sync  (every 5)\n"
         "• replay_age_decay\n• lr_decay  · early_stop",
         COL_SERVER, fontsize=8, fontweight="normal")

    # Aggregator footnote
    rbox(ax, (sx + 0.35, sy + 0.3), 5.6, 1.1,
         "Round-t flow:\n"
         "  receive {z̃_i, y_i}  →  append + age-weighted SGD on θ\n"
         "  every 5 rounds: collect a_i, update T_c, reallocate B",
         "white", fontsize=7.5, ec="#888", lw=0.7)

    # Box for orchestration loop label
    ax.text(sx + sw - 0.3, sy + 0.25, "closed-loop", ha="right",
            fontsize=7.5, color="#A60", style="italic")

    # ===== EVALUATOR =====
    ex, ey, ew, eh = 16.4, 4.7, 3.3, 4.8
    rbox(ax, (ex, ey), ew, eh, "", "white", lw=1.4)
    ax.text(ex + ew / 2, ey + eh - 0.3, "Evaluator  E",
            ha="center", fontsize=11, fontweight="bold", color="#080")
    rbox(ax, (ex + 0.25, ey + 3.0), ew - 0.5, 0.85,
         "CIFAR-100 test\n10K samples", COL_EVAL, fontsize=8.5)
    rbox(ax, (ex + 0.25, ey + 1.95), ew - 0.5, 0.85,
         "predict with θ_t", COL_EVAL, fontsize=8.5)
    rbox(ax, (ex + 0.25, ey + 0.9), ew - 0.5, 0.85,
         "test_acc · macro_F1", COL_EVAL, fontsize=8.5)
    rbox(ax, (ex + 0.25, ey + 0.2), ew - 0.5, 0.55,
         "EarlyStopper (patience=10)", "white",
         fontsize=8, ec="#888")

    # ===== A2A connections =====
    # Client -> A2A (uploads)
    arrow(ax, (cx + cw, 8.6), (bus_x, 8.6), color=ARROW_FWD, lw=1.6)
    arrow(ax, (cx + cw, 7.6), (bus_x, 7.6), color=ARROW_FWD, lw=1.6)
    arrow(ax, (cx + cw, 6.6), (bus_x, 6.6), color=ARROW_FWD, lw=1.6)
    # A2A -> Server
    arrow(ax, (bus_x + bus_w, 8.6), (sx, 8.6), color=ARROW_FWD, lw=1.6)
    arrow(ax, (bus_x + bus_w, 7.6), (sx, 7.6), color=ARROW_FWD, lw=1.6)
    arrow(ax, (bus_x + bus_w, 6.6), (sx, 6.6), color=ARROW_FWD, lw=1.6)
    # Server -> A2A (broadcast head, budget)
    arrow(ax, (sx, 5.7), (bus_x + bus_w, 5.7),
          color=ARROW_REV, lw=1.6, ls="--")
    arrow(ax, (sx, 5.1), (bus_x + bus_w, 5.1),
          color=ARROW_REV, lw=1.6, ls="--")
    # A2A -> Client
    arrow(ax, (bus_x, 5.7), (cx + cw, 5.7),
          color=ARROW_REV, lw=1.6, ls="--")
    arrow(ax, (bus_x, 5.1), (cx + cw, 5.1),
          color=ARROW_REV, lw=1.6, ls="--")

    # Server -> Evaluator
    arrow(ax, (sx + sw, 7.5), (ex, 7.5), color=ARROW_EVAL, lw=1.6)
    ax.text((sx + sw + ex) / 2, 7.7, "head θ_t",
            ha="center", fontsize=8, color=ARROW_EVAL,
            fontweight="bold")

    # ===== BOTTOM: Privacy evaluation =====
    py0, ph = 1.6, 2.5
    ax.add_patch(Rectangle((0.4, py0), 19.2, ph, fc="#FFF7F7",
                            ec="#900", linewidth=1.2,
                            linestyle="--"))
    ax.text(10, py0 + ph - 0.3,
            "Privacy Evaluation  ·  Server-side Decoder Inversion (post-training)",
            ha="center", fontsize=11, fontweight="bold", color="#900")

    pip_y = py0 + 0.4
    pip_h = 1.0
    rbox(ax, (1.0, pip_y), 2.6, pip_h,
         "Client embedding\n z̃ = clip(z) + N(0, σ²)", COL_DP, fontsize=8.5)
    arrow(ax, (3.6, pip_y + pip_h / 2), (4.4, pip_y + pip_h / 2))
    rbox(ax, (4.4, pip_y), 2.8, pip_h,
         "Decoder g_ψ\n(trained on 5K\n decoder pool)", "#E8E8E8",
         fontsize=8.5)
    arrow(ax, (7.2, pip_y + pip_h / 2), (8.0, pip_y + pip_h / 2))
    rbox(ax, (8.0, pip_y), 2.4, pip_h,
         "Reconstruction x̂", COL_DP, fontsize=8.5)
    arrow(ax, (10.4, pip_y + pip_h / 2), (11.2, pip_y + pip_h / 2))
    rbox(ax, (11.2, pip_y), 3.4, pip_h,
         "PSNR(x, x̂)\n= -10·log₁₀ MSE(x, x̂)", "#FFE8E8",
         fontsize=8.5)
    arrow(ax, (14.6, pip_y + pip_h / 2), (15.4, pip_y + pip_h / 2))
    rbox(ax, (15.4, pip_y), 3.6, pip_h,
         "Privacy gain (dB)\nΔ = PSNR_clean − PSNR_σ",
         "#FFD0D0", fontsize=8.5, fontweight="bold")

    # Decoder note
    ax.text(5.8, py0 + 1.7,
            "decoder pool ⊥ federated pool   (server-only auxiliary data)",
            ha="center", fontsize=8, color="#900", style="italic")

    # ===== Legend =====
    leg_y = 0.3
    ax.add_patch(Rectangle((0.4, leg_y), 19.2, 0.8,
                            fc="white", ec="#888", linewidth=0.8))
    items = [
        ("Client Agent", COL_CLIENT),
        ("Server Agent", COL_SERVER),
        ("Evaluator", COL_EVAL),
        ("Privacy / DP", COL_DP),
        ("Replay Buffer", COL_BUFFER),
        ("Data / shared", COL_DATA),
    ]
    arrow_items = [
        ("Upload (client→server)", ARROW_FWD, "-"),
        ("Broadcast (server→client)", ARROW_REV, "--"),
        ("Eval / data flow", ARROW_DATA, ":"),
    ]
    for i, (lab, c) in enumerate(items):
        x0 = 0.7 + i * 1.95
        ax.add_patch(Rectangle((x0, leg_y + 0.25), 0.4, 0.32,
                                fc=c, ec="black"))
        ax.text(x0 + 0.5, leg_y + 0.41, lab, va="center", fontsize=8)

    arrow_x0 = 0.7 + 6 * 1.95
    for i, (lab, c, ls) in enumerate(arrow_items):
        x0 = arrow_x0 + i * 2.6
        a = FancyArrowPatch((x0, leg_y + 0.41), (x0 + 0.7, leg_y + 0.41),
                             arrowstyle="->", mutation_scale=12,
                             color=c, linewidth=1.6, linestyle=ls)
        ax.add_patch(a)
        ax.text(x0 + 0.85, leg_y + 0.41, lab, va="center", fontsize=8)

    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "architecture_overview.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
