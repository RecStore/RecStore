"""DRAM value-store schematics: handle encoding, chunk slabs, zero-copy/RDMA."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import *  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))


def dram_region():
    W, H = 1240, 380
    fig, ax = new_fig(W, H)

    x0, x1, y, h = 40, 1200, 96, 64
    sbox(ax, x0, y, x1 - x0, h, fc="#fafbfc", ec=HEAD, lw=1.5, z=3)
    T(ax, x0, y - 24, "值区：一段连续的进程地址空间（MallocApi 提供，可注册 RDMA）",
      size=11, color=HEAD, w="bold")

    # offset 0 reserved
    sbox(ax, x0, y, 26, h, fc=RED_F, ec=RED, lw=1.5, z=4)
    T(ax, x0 + 13, y + h / 2, "0", size=10.5, color=RED, ha="center",
      va="center", w="bold", z=5)
    vline(ax, x0 + 26, y + h, y + h + 14, RED, 1.1, ls="solid")
    T(ax, x0, y + h + 16, "偏移 0 保留 → handle 0 表示未命中（NONE）", size=10.5,
      color=RED, ha="left", va="top")

    # chunk divisions (1 MiB each, not to scale)
    nchunks = 12
    cw = (x1 - x0 - 26) / nchunks
    for i in range(1, nchunks):
        cx = x0 + 26 + i * cw
        ax.add_line(Line2D([cx, cx], [y, y + h], color=LINE, lw=1.0,
                           linestyle=(0, (3, 3)), zorder=4))
    T(ax, x0 + 26 + 3.5 * cw, y + 22, "1 MiB chunk", size=10.5, color=MUT,
      ha="center", va="center", z=5)
    T(ax, x1, y - 24, "…… 按 1 MiB 切分 ……", size=10, color=MUT, ha="right",
      va="top")
    # a used slot
    sx = x0 + 26 + 6.4 * cw
    sbox(ax, sx, y + 8, cw * 0.5, h - 16, fc=BLUE_F, ec=BLUE, lw=1.4, z=5)
    T(ax, sx + cw * 0.25, y + h / 2, "槽", size=10.5, color=BLUE, ha="center",
      va="center", w="bold", z=6)

    # handle mapping
    tby = 230
    T(ax, 40, tby - 26, "句柄编码：handle = 分配偏移 + 1，解码后才是值区内的字节偏移",
      size=11, color=HEAD, w="bold")
    rbox(ax, 40, tby, 520, 56, fc="white", ec=BLUE, lw=1.5, r=8)
    T(ax, 300, tby + 28, "偏移 0x00123420　→　handle 0x00123421", size=12,
      ha="center", va="center", w="bold", color=BLUE)
    rbox(ax, 600, tby, 520, 56, fc=RED_F, ec=RED, lw=1.5, r=8)
    T(ax, 860, tby + 28, "偏移 0（保留）　→　handle 0 = NONE", size=12,
      ha="center", va="center", w="bold", color=RED)
    save(fig, "dram_value_region", OUT)


def dram_chunk():
    W, H = 1240, 560
    fig, ax = new_fig(W, H)

    # ---- chunk anatomy
    x0, x1, y, h = 40, 1200, 60, 60
    span = x1 - x0
    w1, w2 = 190, 150
    sbox(ax, x0, y, w1, h, fc=ORANGE_F, ec=ORANGE, lw=1.4, z=3)
    T(ax, x0 + w1 / 2, y + 20, "ChunkHeader 8 B", size=11, color=ORANGE,
      ha="center", va="center", w="bold", z=4)
    T(ax, x0 + w1 / 2, y + 42, "slab_size_ 4 B | nr_entries_ 4 B", size=9.5,
      color=ORANGE, ha="center", va="center", z=4)
    sbox(ax, x0 + w1, y, w2, h, fc="#fdf1e0", ec=ORANGE, lw=1.2, z=3)
    T(ax, x0 + w1 + w2 / 2, y + 20, "位图", size=11, color=ORANGE, ha="center",
      va="center", w="bold", z=4)
    T(ax, x0 + w1 + w2 / 2, y + 42, "每槽 1 bit", size=9.5, color=ORANGE,
      ha="center", va="center", z=4)
    sbox(ax, x0 + w1 + w2, y, span - w1 - w2, h, fc="#e8f5ec", ec=GREEN,
         lw=1.4, z=3)
    T(ax, (x0 + w1 + w2 + x1) / 2, y + 30,
      "数据区：nr_entries 个等长槽（nr_entries × slab_size）", size=11.5,
      color=GREEN, ha="center", va="center", w="bold", z=4)
    T(ax, x0, y - 22, "1 个 1 MiB chunk：首次使用时绑定一个尺寸档，之后槽大小固定",
      size=11, color=HEAD, w="bold")

    # ---- size classes
    cy = 176
    T(ax, 40, cy - 24, "尺寸档（载荷 → 槽 = 载荷 + 8 B 长度头）：", size=11,
      color=HEAD, w="bold")
    cls = [(32, 40), (64, 72), (128, 136), (512, 520), (1024, 1032)]
    bw = 190
    for i, (payload, slot) in enumerate(cls):
        bx = 40 + i * (bw + 12)
        hot = payload == 128
        rbox(ax, bx, cy, bw, 78, fc=BLUE_F if hot else "white",
             ec=BLUE if hot else LINE, lw=1.8 if hot else 1.2, r=8)
        T(ax, bx + bw / 2, cy + 18, "载荷 %d B" % payload, size=11,
          color=BLUE if hot else INK, ha="center", va="center",
          w="bold" if hot else "normal")
        T(ax, bx + bw / 2, cy + 40, "槽 %d B = %d + 8" % (slot, payload),
          size=10, color=MUT, ha="center", va="center")
        if hot:
            T(ax, bx + bw / 2, cy + 60,
              "1 MiB / 136 → 7680 槽", size=9.5, color=BLUE, ha="center",
              va="center")

    # ---- one slot, 100 B value
    sy = 300
    T(ax, 40, sy - 24, "写入一条 100 B 的 value：命中 128 档，占用一个 136 B 槽",
      size=11, color=HEAD, w="bold")
    flds = [(8, "", ORANGE_F, ORANGE, ORANGE, 9),
            (100, "100 B 数据", BLUE_F, BLUE, BLUE, 10.5),
            (28, "28 B 空闲", GRAY_F, LINE, MUT, 9)]
    tot = 136.0
    cx = 40
    hdr_cx = None
    for sz, lab, fc, ec, tc, fs in flds:
        wpx = (x1 - 40) * sz / tot
        sbox(ax, cx, sy, wpx, 58, fc=fc, ec=ec, lw=1.3, z=3)
        if lab:
            T(ax, cx + wpx / 2, sy + 29, lab, size=fs, color=tc, ha="center",
              va="center", w="bold" if sz == 100 else "normal", z=4)
        else:
            hdr_cx = cx + wpx / 2
        cx += wpx
    T(ax, 40, sy + 68, "该槽在 chunk 位图中的 1 bit 置 1，回收只需清零这一位",
      size=10.5, color=MUT)
    vline(ax, hdr_cx, sy + 58, sy + 86, ORANGE, 1.2, ls="solid")
    T(ax, hdr_cx, sy + 88, "长度头 8 B = 实际长度 100", size=10, color=ORANGE,
      ha="center", va="top")
    save(fig, "dram_chunk", OUT)


def dram_rdma():
    W, H = 1240, 400
    fig, ax = new_fig(W, H)

    # local consumer
    rbox(ax, 40, 60, 340, 150, fc=BLUE_F, ec=BLUE, lw=1.4, r=10)
    T(ax, 60, 78, "本地读", size=12, color=BLUE, w="bold")
    T(ax, 60, 104, "BatchGet / BatchGetFlat\n直接返回指向槽的视图", size=10.5,
      color=INK, va="top")
    T(ax, 60, 160, "DirectPtr(handle) → 值区真实地址", size=10.5, color=BLUE)

    # value region
    rbox(ax, 470, 60, 380, 150, fc="#fafbfc", ec=HEAD, lw=1.5, r=10)
    T(ax, 660, 78, "值区（同一段地址）", size=12, color=HEAD, ha="center",
      w="bold")
    sbox(ax, 510, 104, 300, 60, fc="#e8f5ec", ec=GREEN, lw=1.3, z=4)
    T(ax, 660, 134, "槽（值字节）", size=11, color=GREEN, ha="center",
      va="center", z=5)
    T(ax, 660, 180, "RDMABackingData / Size", size=10, color=MUT, ha="center")

    # remote
    rbox(ax, 940, 60, 260, 150, fc=ORANGE_F, ec=ORANGE, lw=1.4, r=10)
    T(ax, 1070, 78, "远端 RDMA", size=12, color=ORANGE, ha="center", w="bold")
    T(ax, 1070, 104, "远端 NIC 直接读\n同一段地址", size=10.5, color=INK,
      ha="center", va="top")
    T(ax, 1070, 175, "无需拷贝到中间缓冲", size=10, color=ORANGE, ha="center")

    arrow(ax, 380, 135, 508, 135, color=GREEN, lw=1.8)
    T(ax, 444, 122, "零拷贝视图", size=10, color=GREEN, ha="center", va="bottom",
      w="bold")
    arrow(ax, 812, 135, 940, 135, color=ORANGE, lw=1.8, style="<|-|>")
    T(ax, 876, 122, "RDMA READ", size=10, color=ORANGE, ha="center",
      va="bottom", w="bold")

    save(fig, "dram_rdma", OUT)
