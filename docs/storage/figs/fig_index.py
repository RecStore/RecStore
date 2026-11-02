"""Index-layer schematics (PET hash, SIMD tag match, prefetch, extendible hash)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import *  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))


def pet_hash_bits():
    W, H = 1240, 452
    fig, ax = new_fig(W, H)

    rbox(ax, 30, 18, 120, 46, fc=GRAY_F, ec=LINE)
    T(ax, 90, 41, "key (uint64)", size=11.5, ha="center", va="center")
    arrow(ax, 152, 41, 186, 41)
    rbox(ax, 186, 18, 286, 46, fc="white", ec=HEAD, lw=1.5)
    T(ax, 329, 41, "hash = key × 0xc6a4a7935bd1e995", size=11.5,
      ha="center", va="center")
    arrow(ax, 474, 41, 508, 41)
    rbox(ax, 508, 18, 108, 46, fc=HEAD, ec=HEAD)
    T(ax, 562, 41, "64 位 hash", size=12, color="white", ha="center",
      va="center", w="bold")
    arrow(ax, 562, 64, 562, 96)
    ax.add_line(Line2D([562, 562], [96, 104], color=HEAD, lw=1.6, zorder=4))
    ax.add_line(Line2D([320, 908], [104, 104], color=HEAD, lw=1.6, zorder=4))
    arrow(ax, 320, 104, 320, 128)
    arrow(ax, 908, 104, 908, 128)

    # ---- 桶下标
    rbox(ax, 30, 128, 580, 300, fc=BLUE_F, ec="#c9dbfb", lw=1.4, r=12, z=1)
    T(ax, 52, 144, "① 桶下标 ← hash 低位", size=13, color=BLUE, w="bold")
    bitstrip(ax, 52, 178, 536, 44, [
        {"f": 0.83, "fc": "white", "ec": LINE, "label": "高位", "tc": MUT,
         "size": 10.5},
        {"f": 0.17, "fc": "#d7e4fd", "ec": BLUE, "label": "低位 m 位", "tc": BLUE,
         "size": 10.5, "w": "bold"},
    ])
    T(ax, 52, 228, "bit 63", size=9.5, color=MUT)
    T(ax, 588, 228, "bit 0", size=9.5, color=MUT, ha="right")
    arrow(ax, 528, 222, 528, 258)
    T(ax, 52, 244, "j = hash & (chunk_num − 1)", size=12, color=BLUE, w="bold")
    T(ax, 52, 268, "chunk_table_：每个 F14Chunk = 14 槽 + 14 个 tag（320 B）",
      size=10.5, color=MUT)
    n = 8
    x0, x1, ych, hch = 52, 588, 292, 44
    cw = (x1 - x0) / n
    for i in range(n):
        cx = x0 + i * cw
        hit = i == 3
        sbox(ax, cx, ych, cw - 2, hch, fc="#d7e4fd" if hit else "white",
             ec=BLUE if hit else LINE, lw=1.6 if hit else 1.1, z=3)
        T(ax, cx + (cw - 2) / 2, ych + hch / 2, str(i), size=11,
          color=BLUE if hit else INK, ha="center", va="center",
          w="bold" if hit else "normal", z=4)
    T(ax, 52 + 3 * cw + cw / 2, ych + hch + 6, "chunk[j]", size=10.5, color=BLUE,
      ha="center", va="top", w="bold", z=5)
    T(ax, 588, ych + hch + 6, "……", size=11, color=MUT, ha="right", va="top")

    # ---- 标签
    rbox(ax, 630, 128, 580, 300, fc=ORANGE_F, ec="#f0dcc0", lw=1.4, r=12, z=1)
    T(ax, 652, 144, "② 标签 ← crc32c(hash) 高字节", size=13, color=ORANGE,
      w="bold")
    bitstrip(ax, 652, 178, 536, 44, [
        {"f": 0.25, "fc": "#fbe3c4", "ec": ORANGE, "label": "高位字节 [31:24]",
         "tc": ORANGE, "size": 10.5, "w": "bold"},
        {"f": 0.75, "fc": "white", "ec": LINE, "label": "其余 24 位", "tc": MUT,
         "size": 10.5},
    ])
    T(ax, 652, 228, "bit 31", size=9.5, color=MUT)
    T(ax, 1188, 228, "bit 0", size=9.5, color=MUT, ha="right")
    T(ax, 652, 244, "crc32c(hash)", size=10.5, color=MUT)
    arrow(ax, 720, 222, 720, 276)
    rbox(ax, 652, 276, 536, 42, fc="white", ec=ORANGE, lw=1.5, r=6)
    T(ax, 920, 297, "tag = ((crc32c >> 24) | 0x80) & 0xFF", size=11.5,
      color=ORANGE, ha="center", va="center", w="bold")
    bitstrip(ax, 652, 336, 250, 38, [
        {"f": 0.125, "fc": "#fbe3c4", "ec": ORANGE, "label": "1", "tc": ORANGE,
         "size": 11, "w": "bold"},
        {"f": 0.875, "fc": "white", "ec": LINE, "label": None},
    ])
    T(ax, 652, 378, "bit7", size=9.5, color=MUT)
    T(ax, 902, 378, "bit0", size=9.5, color=MUT, ha="right")
    T(ax, 928, 336,
      "• 每槽 1 B tag" + chr(10) +
      "• SIMD 并行比 14 个 tag" + chr(10) +
      "• 探测步长 = tag × 2 | 1",
      size=10.5, color=INK)
    save(fig, "index_pet_hash_bits", OUT)


def pet_hash_chunk():
    W, H = 1240, 430
    fig, ax = new_fig(W, H)

    y, h = 128, 54
    x0, x1 = 40, 1200
    total = 281.0
    span = x1 - x0

    def seg(b0, b1, label, fc, ec, tc, size=10.5, w="normal"):
        sx = x0 + span * b0 / total
        sw = span * (b1 - b0) / total
        sbox(ax, sx, y, sw, h, fc=fc, ec=ec, lw=1.3, z=3)
        if label:
            T(ax, sx + sw / 2, y + h / 2, label, size=size, color=tc,
              ha="center", va="center", w=w, z=4)

    seg(0, 14, "tags_[14]", "#d7e4fd", BLUE, BLUE, 11, "bold")
    seg(14, 15, None, "#fbe3c4", ORANGE, ORANGE)
    seg(15, 55, "lock_", "#fbe3c4", ORANGE, ORANGE, 11, "bold")
    seg(55, 56, None, "#fbe3c4", ORANGE, ORANGE)
    seg(56, 57, None, "#fbe3c4", ORANGE, ORANGE)
    seg(57, 281, "value_[14]（14 × 16 B）", "#e8f5ec", GREEN, GREEN, 12, "bold")

    # callouts
    T(ax, x0, y - 32, "tags_[14]：14 × 1 B，与槽一一对应", size=11,
      color=BLUE, ha="left", va="center", w="bold")
    vline(ax, x0 + span * 7 / total, y - 16, y, BLUE, 1.2, ls="solid")
    T(ax, x0 + span * 110 / total, y - 30, "元数据 43 B", size=11, color=ORANGE,
      ha="center", va="center", w="bold")
    vline(ax, x0 + span * 110 / total, y - 16, y, ORANGE, 1.2, ls="solid")
    T(ax, x0 + span * 0.70, y - 30, "槽位区：14 个 EntryT", size=11, color=GREEN,
      ha="center", va="center", w="bold")
    vline(ax, x0 + span * 0.70, y - 16, y, GREEN, 1.2, ls="solid")
    T(ax, x0 + span * 169 / total, y + h + 10,
      "size_ 1 B（占用数）　lock_ 40 B　overflow_count_ 1 B（原子）　"
      "is_migrating 1 B（原子）",
      size=10, color=MUT, ha="center", va="top")

    # byte ruler
    ry = y + h + 46
    hline(ax, x0, x1, ry, color=LINE, lw=1.0, ls="solid")
    for b, lab, ha in [(0, "0", "center"), (14, "14", "center"),
                       (58, "58", "center"), (281, "281 B", "center")]:
        bx = x0 + span * b / total
        ax.add_line(Line2D([bx, bx], [ry - 6, ry + 6], color=MUT, lw=1.0,
                           zorder=4))
        T(ax, bx, ry + 8, lab, size=9.5, color=MUT, ha=ha, va="top")

    # entry zoom
    zy = 300
    T(ax, 40, zy - 26, "EntryT（16 B，pack(2)）", size=11, color=GREEN, w="bold")
    n = 4
    zw = (1200 - 40 - 3 * 8) / n
    for i in range(n):
        zx = 40 + i * (zw + 8)
        sbox(ax, zx, zy, zw, 52, fc="white", ec=GREEN, lw=1.3, z=3)
        sbox(ax, zx, zy, zw / 2, 52, fc="#e8f5ec", ec=GREEN, lw=1.0, z=4)
        T(ax, zx + zw / 4, zy + 26, "key", size=10.5, color=GREEN, ha="center",
          va="center", z=5)
        T(ax, zx + zw * 0.75, zy + 26, "handle", size=10.5, color=INK,
          ha="center", va="center", z=5)
        T(ax, zx, zy + 60, "[%d]" % i, size=9.5, color=MUT, va="top")
    T(ax, 40, zy + 96,
      "逻辑 281 B → __aligned__(64) → 320 B / 14 槽（22.9 B/槽）",
      size=10.5, color=MUT)
    save(fig, "index_pet_hash_chunk", OUT)


def pet_hash_simd():
    W, H = 1300, 560
    fig, ax = new_fig(W, H)

    lanes = 16
    x0, lane_w = 60, 74
    nrow = 14
    tags = [0, 224, 61, 0, 143, 224, 0, 0, 200, 0, 0, 0, 0, 0]
    keys = [0, 5, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0]
    qtag = 224

    def lane_x(i):
        return x0 + i * lane_w

    # tag array row
    rbox(ax, 60, 8, 320, 40, fc=GRAY_F, ec=LINE, r=6)
    T(ax, 220, 28, "查询 key = 17　→　tag = 224", size=12, ha="center",
      va="center", w="bold")
    T(ax, 60, 68, "tags_[14]（读入 16 B，后 2 B 由 kFullMask 掩掉）", size=10.5,
      color=BLUE, w="bold")
    ty = 90
    for i in range(lanes):
        if i < nrow:
            hit = tags[i] == qtag
            sbox(ax, lane_x(i), ty, lane_w - 4, 42,
                 fc="#fbe3c4" if hit else "white", ec=ORANGE if hit else LINE,
                 lw=1.8 if hit else 1.1, z=3)
            T(ax, lane_x(i) + (lane_w - 4) / 2, ty + 21, str(tags[i]), size=11,
              color=ORANGE if hit else INK, ha="center", va="center",
              w="bold" if hit else "normal", z=4)
        else:
            sbox(ax, lane_x(i), ty, lane_w - 4, 42, fc="#eeeeee", ec=LINE,
                 lw=1.0, z=3)
            T(ax, lane_x(i) + (lane_w - 4) / 2, ty + 21, "—", size=11, color=MUT,
              ha="center", va="center", z=4)
        T(ax, lane_x(i) + (lane_w - 4) / 2, ty + 48, "slot %d" % i, size=9,
          color=MUT, ha="center", va="top")

    # broadcast row
    by = 196
    T(ax, 60, by - 26, "_mm_set1_epi8(tag)　tag = 224", size=11, color=PURPLE,
      w="bold")
    for i in range(lanes):
        sbox(ax, lane_x(i), by, lane_w - 4, 38, fc=PURPLE_F, ec=PURPLE, lw=1.1,
             z=3)
        T(ax, lane_x(i) + (lane_w - 4) / 2, by + 19, "224", size=10.5,
          color=PURPLE, ha="center", va="center", z=4)

    # compare row
    cy = 262
    for i in range(lanes):
        ok = i < nrow and tags[i] == qtag
        sbox(ax, lane_x(i), cy, lane_w - 4, 34,
             fc="#e8f5ec" if ok else "#f1f4f7", ec=GREEN if ok else LINE,
             lw=1.6 if ok else 1.0, z=3)
        T(ax, lane_x(i) + (lane_w - 4) / 2, cy + 17, "0xFF" if ok else "0x00",
          size=9.5, color=GREEN if ok else MUT, ha="center", va="center",
          w="bold" if ok else "normal", z=4)
    T(ax, 60, cy - 26, "_mm_cmpeq_epi8（逐 lane 比较）", size=11, color=GREEN,
      w="bold")

    # mask row
    my = 328
    T(ax, 60, my - 26, "_mm_movemask_epi8 & 0x3FFF → 14 位掩码", size=11,
      color=TEAL, w="bold")
    for i in range(lanes):
        ok = i < nrow and tags[i] == qtag
        sbox(ax, lane_x(i), my, lane_w - 4, 32,
             fc="#e2f2f4" if ok else "white", ec=TEAL if ok else LINE,
             lw=1.6 if ok else 1.0, z=3)
        T(ax, lane_x(i) + (lane_w - 4) / 2, my + 16, "1" if ok else "0",
          size=11, color=TEAL if ok else MUT, ha="center", va="center",
          w="bold" if ok else "normal", z=4)
    T(ax, 60, my + 40, "掩码中的 1 → 候选槽：slot 1、slot 5", size=10.5,
      color=TEAL, w="bold")

    # slots row
    sy = 430
    T(ax, 60, sy - 26, "只有候选槽读出完整 key 核对；其余 12 个槽跳过 key 读取",
      size=11, color=ORANGE, w="bold")
    for i in range(lanes):
        if i >= nrow:
            continue
        cand = tags[i] == qtag
        fin = cand and keys[i] == 17
        fc = "#d7e4fd" if fin else ("#fdf1e0" if cand else "white")
        ec = BLUE if fin else (ORANGE if cand else LINE)
        sbox(ax, lane_x(i), sy, lane_w - 4, 74, fc=fc, ec=ec,
             lw=1.8 if cand else 1.0, z=3)
        T(ax, lane_x(i) + (lane_w - 4) / 2, sy + 20,
          "key=%d" % keys[i] if cand else "—", size=10,
          color=ORANGE if cand else MUT, ha="center", va="center",
          w="bold" if cand else "normal", z=4)
        if cand:
            verdict = "命中" if fin else "不等"
            T(ax, lane_x(i) + (lane_w - 4) / 2, sy + 48, verdict, size=10,
              color=BLUE if fin else RED, ha="center", va="center", w="bold",
              z=4)
    save(fig, "index_pet_hash_simd", OUT)


def pet_hash_prefetch():
    W, H = 1240, 400
    fig, ax = new_fig(W, H)

    def lane_label(ax, x, y, s, color=INK):
        T(ax, x, y, s, size=11, color=color, ha="right", va="center", w="bold")

    def blk(ax, x, y, w, s, fc, ec, tc):
        sbox(ax, x, y, w, 30, fc=fc, ec=ec, lw=1.2, z=3)
        T(ax, x + w / 2, y + 15, s, size=10, color=tc, ha="center", va="center",
          z=4)

    # ---- 不预取
    T(ax, 40, 30, "不预取：读第 i 个 key 时先等访存，再算，交替串行", size=11.5,
      color=MUT, w="bold")
    x = 150
    y1, y2 = 58, 104
    for i in range(3):
        lane_label(ax, 132, y1 + 15, "访存 i+%d" % i, BLUE)
        blk(ax, x, y1, 150, "取 chunk / 槽", BLUE_F, BLUE, BLUE)
        lane_label(ax, 132, y2 + 15, "计算 i+%d" % i, GREEN)
        blk(ax, x + 155, y2, 130, "tag 比 + key 核对", GREEN_F, GREEN, GREEN)
        x += 300
    ax.add_line(Line2D([40, 1200], [y2 + 40, y2 + 40], color=LINE, lw=1.0,
                       zorder=1))

    # ---- 预取
    T(ax, 40, 210, "预取：算第 i 个 key 之前，先发出第 i+1 个 key 的 HintPrefetch",
      size=11.5, color=ORANGE, w="bold")
    x = 150
    y1, y2 = 238, 284
    for i in range(3):
        lane_label(ax, 132, y1 + 15, "访存 i+%d" % i, BLUE)
        blk(ax, x, y1, 150, "取 chunk / 槽", BLUE_F, BLUE, BLUE)
        lane_label(ax, 132, y2 + 15, "计算 i+%d" % i, GREEN)
        blk(ax, x + 40, y2, 130, "tag 比 + key 核对", GREEN_F, GREEN, GREEN)
        # prefetch for i+1 issued during compute of i
        if i < 2:
            arrow(ax, x + 40, y2 - 4, x + 200, y1 + 34, color=ORANGE, lw=1.4,
                  ms=11, ls=(0, (3, 2)), conn="arc3,rad=-0.25")
            T(ax, x + 128, y1 + 40, "prefetch(i+%d)" % (i + 1), size=9.5,
              color=ORANGE, ha="center", va="top")
        x += 300
    ax.add_line(Line2D([40, 1200], [y2 + 40, y2 + 40], color=LINE, lw=1.0,
                       zorder=1))
    save(fig, "index_pet_hash_prefetch", OUT)


def extendible_hash():
    W, H = 1300, 580
    fig, ax = new_fig(W, H)

    def slots(x, y, w, h, filled, fc, ec, cols=4, rows=4, z=4):
        cw, ch = w / cols, h / rows
        for i in range(cols * rows):
            r, c = divmod(i, cols)
            sx, sy = x + c * cw + 1.5, y + r * ch + 1.5
            used = i < filled
            sbox(ax, sx, sy, cw - 3, ch - 3, fc=fc if used else "white",
                 ec=ec if used else LINE, lw=1.2 if used else 0.9, z=z)

    # ---------------- before ----------------
    T(ax, 40, 24, "分裂前　global_depth = 2（目录 4 项）", size=12.5, color=HEAD,
      w="bold")
    for i in range(4):
        yy = 70 + i * 84
        sbox(ax, 55, yy, 150, 36, fc="white", ec=HEAD, lw=1.3, z=3)
        T(ax, 130, yy + 18, "dir[%d]" % i, size=10.5, ha="center", va="center",
          z=4)
        arrow(ax, 205, yy + 18, 320, yy + 36, color=MUT, lw=1.2, ms=11)
        full = i == 0
        slots(320, yy, 150, 72, 16 if full else 6,
              BLUE_F if not full else "#cfe0fc", BLUE)
        T(ax, 485, yy + 22, "块 B%d" % i, size=10.5, color=BLUE, va="center",
          w="bold")
        T(ax, 485, yy + 44,
          "local_depth = 2　" + ("16/16 满" if full else "部分"),
          size=9.5, color=MUT, va="center")

    # ---------------- after ----------------
    T(ax, 660, 24, "插入第 17 个键 → 分裂　global_depth 2 → 3（目录 8 项）",
      size=12.5, color=HEAD, w="bold")
    xc = [650, 805, 960, 1115]
    for i in range(4):
        sbox(ax, xc[i], 70, 145, 34, fc="#d7e4fd" if i == 0 else "white",
             ec=GREEN if i == 0 else HEAD, lw=1.6 if i == 0 else 1.2, z=3)
        T(ax, xc[i] + 72, 87, "dir[%d]" % i, size=10.5, ha="center", va="center",
          z=4)
        sbox(ax, xc[i], 126, 145, 34, fc="#d7e4fd" if i == 0 else "white",
             ec=GREEN if i == 0 else HEAD, lw=1.6 if i == 0 else 1.2, z=3)
        T(ax, xc[i] + 72, 143, "dir[%d]" % (i + 4), size=10.5, ha="center",
          va="center", z=4)
        col = GREEN if i == 0 else BLUE
        ax.add_line(Line2D([xc[i] + 72, xc[i] + 72], [104, 126], color=col,
                           lw=1.8 if i == 0 else 1.3, zorder=2,
                           linestyle="solid" if i == 0 else (0, (3, 2))))
    T(ax, 1037, 178, "dir[i] 与 dir[i+4] 共享旧块", size=10, color=BLUE,
      ha="center", va="center")
    T(ax, 722, 178, "两块各 local_depth = 3，均分原槽", size=10, color=GREEN,
      ha="center", va="center")

    arrow(ax, 722, 160, 722, 214, color=GREEN, lw=1.4)
    arrow(ax, 722, 160, 870, 214, color=GREEN, lw=1.4, conn="arc3,rad=-0.18")
    slots(650, 214, 145, 130, 8, GREEN_F, GREEN)
    slots(805, 214, 145, 130, 8, GREEN_F, GREEN)
    T(ax, 650, 352, "块 B0a", size=10.5, color=GREEN, w="bold")
    T(ax, 805, 352, "块 B0b", size=10.5, color=GREEN, w="bold")
    sbox(ax, 960, 214, 300, 130, fc=GRAY_F, ec=LINE, lw=1.3, z=3)
    T(ax, 1110, 262, "旧块 B1 / B2 / B3", size=11, color=BLUE, ha="center",
      va="center", w="bold")
    T(ax, 1110, 292, "dir[1]↔dir[5]　dir[2]↔dir[6]　dir[3]↔dir[7]",
      size=10, color=MUT, ha="center", va="center")

    # ---------------- bit rule ----------------
    bitstrip(ax, 40, 400, 1220, 44, [
        {"f": 0.5, "fc": "white", "ec": LINE,
         "label": "key_hash = std::_Hash_bytes(key, 8) 的高位（不参与索引）",
         "tc": MUT, "size": 10.5},
        {"f": 0.5, "fc": "#d7e4fd", "ec": BLUE,
         "label": "低 global_depth 位 → 目录下标 x = key_hash & (2^global_depth − 1)",
         "tc": BLUE, "size": 11, "w": "bold"},
    ])
    T(ax, 40, 454, "bit 63", size=9.5, color=MUT)
    T(ax, 1260, 454, "bit 0", size=9.5, color=MUT, ha="right")
    save(fig, "index_extendible_hash", OUT)
