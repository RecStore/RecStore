import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import *  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))


def fig_pet_hash_bits():
    W, H = 1300, 700
    fig, ax = new_fig(W, H)
    T(ax, 40, 28, "PET 哈希 · 一次哈希，同时给出「在哪个桶」和「和哪些标签比」",
      size=17, w="bold")
    T(ax, 40, 56,
      "hash = key × 0xc6a4a7935bd1e995（uint64）；桶下标取 hash 低位，标签取 crc32c(hash) 的高字节",
      size=11.5, color=MUT)

    rbox(ax, 40, 90, 130, 48, fc=GRAY_F, ec=LINE)
    T(ax, 105, 114, "key" + chr(10) + "(uint64)", size=11.5, ha="center", va="center")
    arrow(ax, 172, 114, 208, 114)
    rbox(ax, 208, 90, 300, 48, fc="white", ec=HEAD, lw=1.5)
    T(ax, 358, 114, "hash = key × 0xc6a4a7935bd1e995", size=11.5,
      ha="center", va="center")
    arrow(ax, 510, 114, 546, 114)
    rbox(ax, 546, 90, 120, 48, fc=HEAD, ec=HEAD)
    T(ax, 606, 114, "64 位 hash", size=12, color="white",
      ha="center", va="center", w="bold")

    ax.add_line(Line2D([606, 606], [138, 158], color=HEAD, lw=1.6, zorder=4))
    ax.add_line(Line2D([345, 953], [158, 158], color=HEAD, lw=1.6, zorder=4))
    arrow(ax, 345, 158, 345, 190)
    arrow(ax, 953, 158, 953, 190)

    rbox(ax, 40, 190, 610, 400, fc=BLUE_F, ec="#c9dbfb", lw=1.4, r=12, z=1)
    T(ax, 66, 208, "① 桶下标 ← hash 低位", size=13.5, color=BLUE, w="bold")
    bitstrip(ax, 66, 248, 558, 46, [
        {"f": 0.83, "fc": "white", "ec": LINE, "label": "hash 高位（不参与选桶）",
         "tc": MUT, "size": 10.5},
        {"f": 0.17, "fc": "#d7e4fd", "ec": BLUE, "label": "低位 m 位", "tc": BLUE,
         "size": 10.5, "w": "bold"},
    ])
    T(ax, 66, 302, "bit 63", size=9.5, color=MUT)
    T(ax, 624, 302, "bit 0", size=9.5, color=MUT, ha="right")
    arrow(ax, 558, 296, 558, 356)
    T(ax, 66, 320, "j = hash & (chunk_num − 1)　　m = log2(chunk_num)，chunk_num 为 2 的幂",
      size=11.5, color=BLUE, w="bold")
    T(ax, 66, 344, "chunk_table_：每个 chunk 一个 F14Chunk（14 槽 + 14 个 tag）",
      size=10.5, color=MUT)
    n = 8
    x0, x1, ych, hch = 66, 624, 366, 46
    cw = (x1 - x0) / n
    for i in range(n):
        cx = x0 + i * cw
        hit = i == 3
        sbox(ax, cx, ych, cw - 2, hch, fc="#d7e4fd" if hit else "white",
             ec=BLUE if hit else LINE, lw=1.6 if hit else 1.1, z=3)
        T(ax, cx + (cw - 2) / 2, ych + hch / 2, str(i), size=11,
          color=BLUE if hit else INK, ha="center", va="center",
          w="bold" if hit else "normal", z=4)
    T(ax, 66 + 3 * cw + cw / 2, ych + hch + 6, "chunk[j]", size=10.5, color=BLUE,
      ha="center", va="top", w="bold", z=5)
    T(ax, 626, ych + hch + 6, "…… 共 chunk_num 个 ……", size=10, color=MUT,
      ha="right", va="top")
    T(ax, 66, 446,
      "1e6 记录 → chunk_num = 131072（m = 17）" + chr(10) +
      "chunk_table_ ≈ 131072 × 320 B = 40 MiB → 41.9 B/键" + chr(10) +
      "14 个 key 的 hash 低位相同就落进同一 chunk，再由 tag 区分",
      size=10.5, color=MUT)

    rbox(ax, 660, 190, 600, 400, fc=ORANGE_F, ec="#f0dcc0", lw=1.4, r=12, z=1)
    T(ax, 686, 208, "② 标签 ← crc32c(hash) 的高字节", size=13.5, color=ORANGE,
      w="bold")
    bitstrip(ax, 686, 248, 548, 46, [
        {"f": 0.25, "fc": "#fbe3c4", "ec": ORANGE, "label": "高位字节 [31:24]",
         "tc": ORANGE, "size": 10.5, "w": "bold"},
        {"f": 0.75, "fc": "white", "ec": LINE, "label": "crc32c 其余 24 位",
         "tc": MUT, "size": 10.5},
    ])
    T(ax, 686, 302, "bit 31", size=9.5, color=MUT)
    T(ax, 1234, 302, "bit 0", size=9.5, color=MUT, ha="right")
    T(ax, 686, 320, "crc32c(hash)：32 位", size=10.5, color=MUT)
    arrow(ax, 754, 296, 754, 352)
    rbox(ax, 686, 352, 548, 44, fc="white", ec=ORANGE, lw=1.5, r=6)
    T(ax, 960, 374, "tag = ((crc32c >> 24) | 0x80) & 0xFF", size=11.5,
      color=ORANGE, ha="center", va="center", w="bold")
    bitstrip(ax, 686, 414, 268, 40, [
        {"f": 0.125, "fc": "#fbe3c4", "ec": ORANGE, "label": "1", "tc": ORANGE,
         "size": 11, "w": "bold"},
        {"f": 0.875, "fc": "white", "ec": LINE, "label": None},
    ])
    T(ax, 686, 456, "bit7", size=9.5, color=MUT)
    T(ax, 954, 456, "bit0", size=9.5, color=MUT, ha="right")
    T(ax, 686, 472, "bit7 恒置 1 → tag ∈ [128, 255]，不会与空槽 tag = 0 混淆",
      size=10.5, color=MUT)
    T(ax, 1010, 412,
      "• 每槽 1 B tag" + chr(10) +
      "• SIMD 并行比较 14 个 tag" + chr(10) +
      "• 探测步长 = tag × 2 | 1",
      size=10.5, color=INK)

    T(ax, 40, 628,
      "tag 只做筛选：标签相等仅说明该槽是候选，最终判定要读出槽内完整的 64 位 key 再逐字节核对。",
      size=11, color=MUT, style="italic")
    T(ax, 40, 652,
      "「14 个 tag 一次比完」把绝大多数不匹配的槽挡在 key 比较之前，省掉逐槽读 key 的开销。",
      size=11, color=MUT, style="italic")
    save(fig, "index_pet_hash_bits", OUT)


if __name__ == "__main__":
    fig_pet_hash_bits()
