"""
LCM-SPP Pruning Mechanism — Publication-quality English figure
Two panels:
  (a) LCM set-enumeration tree with colour-coded pruning outcomes
  (b) Per-node decision flowchart  (code order: support → v-bound → u-bound)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      200,
    "savefig.dpi":     300,
})

# ─── Colour palette ───────────────────────────────────────────────────────────
C_VISIT   = "#3E7CB1"   # steel blue   – visited
C_SELECT  = "#2E8B57"   # sea green    – added to candidate set
C_VPRUNE  = "#C0392B"   # deep red     – v-bound prune
C_USCREEN = "#D07000"   # amber        – u-bound screen
C_WS      = "#6B3FA0"   # violet       – WS-removed
C_GHOST   = "#C0C0C0"   # silver       – pruned subtree placeholder
C_FLOW_H  = "#1F4E79"   # header / start / recurse box
C_FLOW_M  = "#2E75B6"   # mid-level box (compute child_tid)
PANEL_BG  = "#F5F7FA"   # soft blue-grey panel background
SEP_COL   = "#C8CBD0"   # panel separator colour
WHITE     = "#FFFFFF"
BG        = "#FFFFFF"

# ─── Figure & panels ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10), facecolor=BG)

# thin vertical separator
fig.add_artist(plt.Line2D(
    [0.512, 0.512], [0.02, 0.98],
    transform=fig.transFigure, color=SEP_COL, lw=1.3, zorder=10))

ax_tree = fig.add_axes([0.010, 0.025, 0.494, 0.950])
ax_flow = fig.add_axes([0.522, 0.025, 0.468, 0.950])

for ax in (ax_tree, ax_flow):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
#  PANEL (a)  LCM Set-Enumeration Tree
# ════════════════════════════════════════════════════════════════════════════
ax = ax_tree
ax.set_xlim(-0.8, 11.8)
ax.set_ylim(-1.6, 8.4)
ax.axis("off")

R = 0.44   # node radius

nodes = {
    "root":  ( 5.50, 7.50, "$\\emptyset$", C_VISIT,   "visit"),
    "n1":    ( 2.00, 5.90, "{1}",      C_VISIT,   "visit"),
    "n2":    ( 5.50, 5.90, "{2}",      C_VISIT,   "visit"),
    "n3":    ( 9.00, 5.90, "{3}",      C_VISIT,   "visit"),
    "n12":   ( 0.50, 4.30, "{1,2}",   C_VPRUNE,  "vprune"),
    "n13":   ( 2.00, 4.30, "{1,3}",   C_USCREEN, "uscreen"),
    "n14":   ( 3.50, 4.30, "{1,4}",   C_SELECT,  "select"),
    "n23":   ( 4.80, 4.30, "{2,3}",   C_SELECT,  "select"),
    "n25":   ( 6.20, 4.30, "{2,5}",   C_USCREEN, "uscreen"),
    "n34":   ( 7.80, 4.30, "{3,4}",   C_VPRUNE,  "vprune"),
    "n35":   ( 9.80, 4.30, "{3,5}",   C_WS,      "ws"),
    "n134":  ( 2.00, 2.70, "{1,3,4}", C_VPRUNE,  "vprune"),
    "n235":  ( 4.20, 2.70, "{2,3,5}", C_SELECT,  "select"),
    "n236":  ( 5.80, 2.70, "{2,3,6}", C_VPRUNE,  "vprune"),
}

edges = [
    ("root","n1"),  ("root","n2"),  ("root","n3"),
    ("n1","n12"),   ("n1","n13"),   ("n1","n14"),
    ("n2","n23"),   ("n2","n25"),
    ("n3","n34"),   ("n3","n35"),
    ("n13","n134"),
    ("n23","n235"), ("n23","n236"),
]

E_COLOUR = {
    "visit":   "#AAAAAA", "select":  "#AAAAAA",
    "vprune":  "#E07060", "uscreen": "#BBBBBB", "ws": "#A888CC",
}

def draw_edge(ax, k1, k2):
    x1, y1 = nodes[k1][:2];  x2, y2 = nodes[k2][:2]
    ntype   = nodes[k2][4];   col     = E_COLOUR[ntype]
    ls      = "--" if ntype == "uscreen" else "-"
    dx, dy  = x2-x1, y2-y1;  d = np.hypot(dx, dy)
    ux, uy  = dx/d, dy/d
    ax.annotate("", xy=(x2-ux*R, y2-uy*R), xytext=(x1+ux*R, y1+uy*R),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                linestyle=ls, mutation_scale=11), zorder=1)

for e in edges:
    draw_edge(ax, *e)

def pruned_tri(ax, cx, cy_bot, h=0.70):
    pts = [[cx-.48, cy_bot], [cx+.48, cy_bot], [cx, cy_bot-h]]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#D8D8D8",
                         edgecolor="#AAAAAA", lw=0.9, ls="--", zorder=1, alpha=0.80))
    ax.text(cx, cy_bot - h*0.53, "pruned\nsubtree",
            ha="center", va="center", fontsize=5.8, color="#888888", zorder=2)

for key in ("n12", "n34", "n134", "n236"):
    cx, cy = nodes[key][:2]
    pruned_tri(ax, cx, cy - R - 0.04)

def draw_node(ax, cx, cy, label, colour):
    # drop shadow
    ax.add_patch(Circle((cx+0.07, cy-0.07), R, facecolor="#AAAAAA",
                         edgecolor="none", zorder=2, alpha=0.35))
    ax.add_patch(Circle((cx, cy), R, facecolor=colour, edgecolor=WHITE,
                         linewidth=2.2, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=WHITE, zorder=4)

for key, (cx, cy, lbl, col, _) in nodes.items():
    draw_node(ax, cx, cy, lbl, col)

def callout(ax, tx, ty, nx, ny, text, fc, ec, tc, fs=7.0):
    dx, dy = tx-nx, ty-ny;  dist = np.hypot(dx, dy)
    sx, sy = nx + dx/dist*R, ny + dy/dist*R
    ax.plot([sx, tx], [sy, ty], color=ec, lw=0.85, ls=(0, (4, 3)), zorder=2)
    ax.text(tx, ty, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=5, multialignment="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=fc,
                      edgecolor=ec, linewidth=1.1, alpha=0.97))

callout(ax, -0.30, 3.00, *nodes["n12"][:2],
        "v-bound < λ:\nentire subtree pruned",    "#FDECEA", C_VPRUNE,  "#7A0000")
callout(ax,  1.10, 5.25, *nodes["n13"][:2],
        "u-bound < λ:\nno candidate,\nbut recurse", "#FFF3E0", C_USCREEN, "#7A4B00")
callout(ax,  3.50, 3.05, *nodes["n14"][:2],
        "u-bound ≥ λ:\nadded to\ncandidate set",  "#EAF5EC", C_SELECT,  "#1A5C30")
callout(ax,  9.80, 3.00, *nodes["n35"][:2],
        "Added initially;\nWS-removed\n(weight $\\to$ 0)", "#F2EEF9", C_WS, "#42107A")

# v-bound formula (top-right)
ax.text(9.95, 7.50,
        r"v-bound$(S) = |\sum_{i\in S}\alpha_i| + r\cdot\|x_S^c\|$"
        "\n" r"(+2ref: $\min(\mathrm{v}_1,\mathrm{v}_2)$ using prev. $\lambda$ soln.)",
        ha="center", va="center", fontsize=7.5, color="#1A205E", zorder=5,
        multialignment="center",
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#EEF2FF",
                  edgecolor="#8090C8", linewidth=1.1, alpha=0.97))

# Panel label & title
ax.text(-0.65, 8.35, "(a)", fontsize=13, fontweight="bold",
        va="top", color="#1A1A2E")
ax.text( 0.10, 8.35, "LCM Set-Enumeration Tree with SPP Pruning",
         fontsize=11, fontweight="bold", va="top", color="#1A1A2E")

legend_items = [
    mpatches.Patch(fc=C_VISIT,   ec=WHITE, label="Visited (no action yet)"),
    mpatches.Patch(fc=C_SELECT,  ec=WHITE, label="Added to candidate set (u-bound ≥ λ)"),
    mpatches.Patch(fc=C_USCREEN, ec=WHITE, label="u-screened: skip candidate, still recurse"),
    mpatches.Patch(fc=C_VPRUNE,  ec=WHITE, label="v-pruned: entire subtree eliminated"),
    mpatches.Patch(fc=C_WS,      ec=WHITE, label=r"WS-removed: weight $\to$ 0 between rounds"),
    mpatches.Patch(fc=C_GHOST,   ec="#AAAAAA", label="Eliminated subtree (placeholder)"),
]
leg = ax.legend(handles=legend_items, loc="lower left",
                bbox_to_anchor=(0.00, 0.00), ncol=2,
                framealpha=0.97, facecolor=WHITE, edgecolor=SEP_COL,
                fontsize=7.5, handlelength=1.4, columnspacing=1.0)
leg.get_frame().set_linewidth(1.0)


# ════════════════════════════════════════════════════════════════════════════
#  PANEL (b)  Per-Node Decision Flowchart  (code order: support → v-bound → u-bound)
# ════════════════════════════════════════════════════════════════════════════
ax = ax_flow
ax.set_xlim(0, 10)
ax.set_ylim(0.4, 13.1)
ax.axis("off")

ax.text(0.15, 12.95, "(b)", fontsize=13, fontweight="bold",
        va="top", color="#1A1A2E")
ax.text(1.00, 12.95, "Per-Node Decision Logic",
        fontsize=11, fontweight="bold", va="top", color="#1A1A2E")

CX  = 4.65   # main-column centre x
BW  = 4.50   # process-box full width
BHH = 0.310  # process-box half-height
DW2 = 2.30   # diamond half-width
DHH = 0.53   # diamond half-height

def proc(ax, cx, cy, label, fc, tc=WHITE, fs=8.5, bw=None, bhh=None):
    bw  = bw  or BW
    bhh = bhh or BHH
    # shadow
    ax.add_patch(FancyBboxPatch(
        (cx - bw/2 + 0.08, cy - bhh - 0.08), bw, 2*bhh,
        boxstyle="round,pad=0.07", facecolor="#AAAAAA",
        edgecolor="none", zorder=2, alpha=0.30))
    ax.add_patch(FancyBboxPatch(
        (cx - bw/2, cy - bhh), bw, 2*bhh,
        boxstyle="round,pad=0.07", facecolor=fc,
        edgecolor=WHITE, linewidth=1.8, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=4, multialignment="center")
    return dict(top=cy+bhh, bot=cy-bhh,
                left=cx-bw/2, right=cx+bw/2, cx=cx, cy=cy)

def dmd(ax, cx, cy, label, fc="#EEEEEE", tc="#222222", fs=8.0):
    # shadow
    sp = [[cx+0.08, cy+DHH-0.08], [cx+DW2+0.08, cy-0.08],
          [cx+0.08, cy-DHH-0.08], [cx-DW2+0.08, cy-0.08]]
    ax.add_patch(Polygon(sp, closed=True, facecolor="#AAAAAA",
                         edgecolor="none", zorder=2, alpha=0.30))
    pts = [[cx, cy+DHH], [cx+DW2, cy], [cx, cy-DHH], [cx-DW2, cy]]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                         edgecolor=WHITE, linewidth=1.8, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=4, multialignment="center")
    return dict(top=cy+DHH, bot=cy-DHH,
                left=cx-DW2, right=cx+DW2, cx=cx, cy=cy)

def varr(ax, x, y1, y2, col="#555555", lw=1.6):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=13), zorder=2)

def harr(ax, x1, x2, y, col="#555555", lw=1.6):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=13), zorder=2)

def yes_lbl(ax, x, y):
    ax.text(x + 0.14, y + 0.07, "Yes", ha="left", va="bottom",
            fontsize=7.5, color="#AA2222", fontweight="bold")

def no_lbl(ax, x, y):
    ax.text(x - 0.14, y - 0.07, "No", ha="right", va="top",
            fontsize=7.5, color="#226622", fontweight="bold")

# ── Flowchart nodes (top → bottom): support → v-bound → u-bound ──────────────
b_start  = proc(ax, CX, 12.15, "Enumerate node: extend prefix by item j",     fc=C_FLOW_H)
b_ctid   = proc(ax, CX, 11.10, r"child_tid $\leftarrow$ tid $\cap$ TID[$j$]", fc=C_FLOW_M)
d_sup    = dmd (ax, CX,  9.95, "support(child_tid) < θ ?",                     fc="#EBEBEB", tc="#2A2A2A")
d_vbound = dmd (ax, CX,  8.50, "v_bound(child_tid) < λ ?",                     fc="#FDECEA", tc="#5A0000")
d_ubound = dmd (ax, CX,  7.05, "u_bound(child_tid) < λ ?",                     fc="#FFF3E0", tc="#5A3000")
b_add    = proc(ax, CX,  5.78, "Add to candidate set",                          fc=C_SELECT)
b_recurs = proc(ax, CX,  4.68, "Recurse on child nodes (depth-first)",          fc=C_FLOW_H)

# ── Main-column arrows ───────────────────────────────────────────────────────
varr(ax, CX, b_start["bot"],  b_ctid["top"])
varr(ax, CX, b_ctid["bot"],   d_sup["top"])
no_lbl(ax, CX, (b_ctid["bot"] + d_sup["top"]) / 2)
varr(ax, CX, d_sup["bot"],    d_vbound["top"])
no_lbl(ax, CX, (d_sup["bot"] + d_vbound["top"]) / 2)
varr(ax, CX, d_vbound["bot"], d_ubound["top"])
no_lbl(ax, CX, (d_vbound["bot"] + d_ubound["top"]) / 2)
varr(ax, CX, d_ubound["bot"], b_add["top"])
no_lbl(ax, CX, (d_ubound["bot"] + b_add["top"]) / 2)
varr(ax, CX, b_add["bot"],    b_recurs["top"])

# ── Right-side outcome boxes ──────────────────────────────────────────────────
RX  = CX + DW2 + 1.00
OBW = 1.72

b_disc = proc(ax, RX, d_sup["cy"],    "Discard\n(support)",            fc="#888888", fs=7.5, bw=OBW)
b_vprn = proc(ax, RX, d_vbound["cy"], "Prune entire\nsubtree",         fc=C_VPRUNE,  fs=7.5, bw=OBW)
b_uscr = proc(ax, RX, d_ubound["cy"], "Skip candidate;\nstill recurse", fc=C_USCREEN, fs=7.5, bw=OBW)

harr(ax, d_sup["right"],    b_disc["left"], d_sup["cy"],    col="#888888")
yes_lbl(ax, d_sup["right"],    d_sup["cy"])
harr(ax, d_vbound["right"], b_vprn["left"], d_vbound["cy"], col=C_VPRUNE)
yes_lbl(ax, d_vbound["right"], d_vbound["cy"])
harr(ax, d_ubound["right"], b_uscr["left"], d_ubound["cy"], col=C_USCREEN)
yes_lbl(ax, d_ubound["right"], d_ubound["cy"])

# ── Dashed detour: u-screened → bypasses "Add" → still recurses ──────────────
DX = b_uscr["right"] + 0.30
ax.plot([b_uscr["cx"],  DX],            [b_uscr["bot"],   b_uscr["bot"]],
        color=C_USCREEN, lw=1.3, ls="--", zorder=2)
ax.plot([DX, DX],                       [b_uscr["bot"],   b_recurs["cy"]],
        color=C_USCREEN, lw=1.3, ls="--", zorder=2)
ax.plot([DX, b_recurs["right"] + 0.06], [b_recurs["cy"],  b_recurs["cy"]],
        color=C_USCREEN, lw=1.3, ls="--", zorder=2)
ax.annotate("", xy=(b_recurs["right"], b_recurs["cy"]),
            xytext=(b_recurs["right"] + 0.06, b_recurs["cy"]),
            arrowprops=dict(arrowstyle="-|>", color=C_USCREEN,
                            lw=1.3, mutation_scale=11), zorder=2)
# label on dashed bypass
ax.text(DX + 0.10, (b_uscr["bot"] + b_recurs["cy"]) / 2,
        "bypass\nadd step", ha="left", va="center",
        fontsize=6.5, color=C_USCREEN, fontstyle="italic")

# ── Formula annotations (left of diamonds) ────────────────────────────────────
# v-bound formula (left of d_vbound)
ax.text(CX - DW2 - 0.18, d_vbound["cy"],
        "v_bound$(S)$:\n"
        r"$|\sum_{i\in S}\alpha_i| + r\cdot\|x_S^c\|$" "\n"
        r"(+2ref: $\min(\mathrm{v}_1,\mathrm{v}_2)$)",
        ha="right", va="center", fontsize=7.5, color="#7A0000",
        multialignment="right",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF5F5",
                  edgecolor=C_VPRUNE, linewidth=1.0, alpha=0.95))

# u-bound formula (left of d_ubound)
ax.text(CX - DW2 - 0.18, d_ubound["cy"],
        "u_bound$(S)$:\n"
        r"$|x_S^T\alpha| + r_u\cdot\|x_S^c\|$" "\n"
        r"(+2ref: $\min(\mathrm{u}_1,\mathrm{u}_2)$)",
        ha="right", va="center", fontsize=7.5, color="#7A4B00",
        multialignment="right",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFBF0",
                  edgecolor=C_USCREEN, linewidth=1.0, alpha=0.95))

# ── WS + 2ref note ────────────────────────────────────────────────────────────
ws_y = b_recurs["cy"] - 1.18
ax.add_patch(FancyBboxPatch(
    (0.28, ws_y - 0.44), 9.44, 0.82,
    boxstyle="round,pad=0.14", facecolor="#F0E8F8",
    edgecolor=C_WS, linewidth=1.6, zorder=3))
ax.text(5.00, ws_y,
        r"Working Set (WS): after each FISTA round, remove patterns with $|w_k| < \varepsilon$"
        r"   $\cdot$   "
        r"2ref: use previous $\lambda$ solution $(\alpha_2,\, r_2)$ to tighten both bounds",
        ha="center", va="center", fontsize=7.5, color="#320050",
        fontweight="bold", zorder=4, multialignment="center")

# ─── Super-title ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.998,
         "LCM-SPP: Safe Pattern Pruning in the Set-Enumeration Tree",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")

# ─── Save ─────────────────────────────────────────────────────────────────────
out = "pruning_diagram_en"
plt.savefig(out + ".pdf", bbox_inches="tight", facecolor=BG)
plt.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved  {out}.pdf  and  {out}.png")
