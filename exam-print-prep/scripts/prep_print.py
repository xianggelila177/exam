#!/usr/bin/env python3
"""
prep_print.py — 把试卷/讲义的截图或拍照，批量处理成可直接打印的稿件。

子命令：

  survey   给一批图生成缩略图联表 + 分页页脚拼贴 + 每张图的裁剪诊断，
           供 Claude 用眼睛看一遍，判断分组和裁剪是否正确。
  build    裁黑边 → 升/降采样到目标尺寸 → 锐化 → 对比增强 → 出 JPG + 合并 PDF。
  verify   对 build 的输出做视觉自检：生成成品缩略图、尺寸/边缘统计。
  pack     把若干输出目录打成一个 zip。
  split    把横屏双页图按中缝切成左右两页。

设计要点：
- 裁剪用"行/列亮度扫描"找纸面，能同时干掉播放器黑边、状态栏、页码条。
- 所有增强参数都可从命令行调，因为不同来源的截图脏的程度差很多。
- 脚本只做确定性的像素活儿；"哪几页是一套""哪套带答案"这类判断留给 Claude。
- 依赖检查放在入口，缺 Pillow/NumPy 时明确报错，不尝试在脚本里安装。
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on runtime
    np = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover - depends on runtime
    Image = None

if Image is not None:
    Image.MAX_IMAGE_PIXELS = None

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
FOOTER_STRIP_PAGES = 12  # 页脚长图分批，避免单张图无限纵向拼接


def require_deps():
    missing = []
    if np is None:
        missing.append("NumPy")
    if Image is None:
        missing.append("Pillow")
    if missing:
        sys.exit(
            "当前运行环境缺少 %s，无法执行此 Skill。\n"
            "请先安装依赖（例如 python3-pil python3-numpy），或由宿主环境注入依赖。"
            % " 和 ".join(missing)
        )


# ---------------------------------------------------------------- 工具


def natural_key(path):
    """按文件名里的数字自然排序，保证 2.jpg 排在 10.jpg 前面。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def path_sort_key(path):
    """排序键：先按父目录分组，再按文件名自然排序。

    修复旧版只按 basename 排序导致的“原卷/解析同名文件交错”问题。
    """
    path = os.path.normpath(path)
    return (os.path.dirname(path).lower(), natural_key(path), path.lower())


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dedupe_files(files):
    """按 MD5 去掉完全相同的图片，保留按目录/页序排在最前的一份。"""
    seen = {}
    out = []
    for f in files:
        digest = md5_file(f)
        if digest not in seen:
            seen[digest] = f
            out.append(f)
        # 如果完全相同且已在前面保留，跳过；可在这里记录 duplicate 映射，
        # 但当前保持 CLI 输出简单，只返回保留列表。
    return out


def collect(paths, dedupe=False):
    """接受文件、目录或 zip，统一返回排好序的图片路径列表。"""
    require_deps()
    raw = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                if "__MACOSX" in root:
                    continue
                for f in files:
                    if f.lower().endswith(EXTS) and not f.startswith("._"):
                        raw.append(os.path.join(root, f))
        elif p.lower().endswith(".zip"):
            dest = tempfile.mkdtemp(prefix="prep_print_")
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            raw += collect([dest], dedupe=False)
        elif p.lower().endswith(EXTS):
            raw.append(p)

    # 去重路径，保留首次出现；再按目录+文件名自然排序。
    seen_paths = set()
    out = []
    for p in raw:
        norm = os.path.normpath(os.path.abspath(p))
        if norm not in seen_paths:
            seen_paths.add(norm)
            out.append(norm)
    out.sort(key=path_sort_key)

    if dedupe:
        out = dedupe_files(out)
    return out


def find_page(im, thresh=150, margin=0):
    """
    找纸面的外接框。

    思路：纸是亮的，播放器黑边/状态栏/页码条是暗的。先按行求平均亮度，
    留下亮的行；再在这些行里按列求平均亮度，留下亮的列。这比"逐像素找非黑"
    稳，因为页码条上的白字不会把整行的平均亮度拉上去。

    返回 (left, top, right, bottom)；找不到纸面就返回整图。
    """
    a = np.asarray(im.convert("L"), dtype=np.float32)
    rows = np.where(a.mean(axis=1) > thresh)[0]
    if rows.size == 0:
        return (0, 0, im.width, im.height)
    t, b = int(rows[0]), int(rows[-1]) + 1
    cols = np.where(a[t:b].mean(axis=0) > thresh)[0]
    if cols.size == 0:
        return (0, t, im.width, b)
    l, r = int(cols[0]), int(cols[-1]) + 1
    if margin:
        l, t = max(0, l + margin), max(0, t + margin)
        r, b = min(im.width, r - margin), min(im.height, b - margin)
    return (l, t, r, b)


def apply_whitepoint(im, whitepoint):
    """把亮度 >= whitepoint 的像素推成纯白，低于阈值的像素保持不变。"""
    if whitepoint >= 255:
        return im
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    gray = a.mean(axis=2)
    mask = gray >= whitepoint
    if mask.any():
        a[mask] = 255.0
    return Image.fromarray(a.astype(np.uint8))


def apply_ink_save(im, ink_limit=0.45, radius=15):
    """省墨模式：把大面积深色区域调淡，同时保留细笔画/文字的黑色。

    做法是先高斯模糊定位“大面积深色”，再只对这些区域做墨量限制；
    细小的文字因为模糊后平均亮度变高，不会被误判为大面积深色。
    """
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    g = arr.mean(axis=2)
    blurred = np.asarray(
        Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    )
    large_dark = blurred < 128
    if not large_dark.any():
        return im
    # 大面积深色区域映射到指定墨量（保留暗部形状，颜色转灰以省墨）；
    # 非大面积深色区域保持原样，细笔画/彩色内容不受影响。
    dark_gray = 255.0 - (255.0 - g) * ink_limit
    mask = np.repeat(large_dark[:, :, None], 3, axis=2)
    arr = np.where(mask, np.repeat(dark_gray[:, :, None], 3, axis=2), arr)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def enhance(im, target_w, sharpen, contrast, radius, threshold, grayscale,
            whitepoint, fit_a4, page_margin, ink_save=False, ink_limit=0.45):
    """裁边后处理：缩放 → 提白 → USM 锐化 → 对比增强 → 可选 A4 画布。

    - 不启用 fit_a4 时，target_w 是精确的目标宽度，比它大或小都会缩放到该宽度。
    - 启用 fit_a4 时，先保持长宽比缩放到 A4 画布内，再居中贴到白底画布。
    """
    if not fit_a4 and target_w:
        scale = target_w / im.width
        im = im.resize((round(im.width * scale), round(im.height * scale)),
                       Image.LANCZOS)
    if whitepoint < 255:
        im = apply_whitepoint(im, whitepoint)
    if sharpen:
        im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen,
                                               threshold=threshold))
    if contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(contrast)
    if grayscale:
        im = im.convert("L").convert("RGB")
    if ink_save:
        im = apply_ink_save(im, ink_limit)

    if fit_a4:
        canvas_w = target_w or 2480
        canvas_h = math.ceil(canvas_w * math.sqrt(2))
        margin = max(0, page_margin)
        usable_w = max(1, canvas_w - 2 * margin)
        usable_h = max(1, canvas_h - 2 * margin)
        scale = min(usable_w / im.width, usable_h / im.height)
        if scale != 1.0:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        canvas.paste(im, ((canvas_w - im.width) // 2,
                          (canvas_h - im.height) // 2))
        im = canvas
    return im


def edge_warnings(im, edge_min):
    """检查处理后图片四边平均亮度，低于阈值说明可能有黑边残留。"""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    edges = [a[0].mean(), a[-1].mean(), a[:, 0].mean(), a[:, -1].mean()]
    return [round(e) for e in edges], min(edges)


# ---------------------------------------------------------------- survey


def cmd_survey(args):
    files = collect(args.inputs, dedupe=args.dedupe)
    if not files:
        sys.exit("没有找到图片")
    d = os.path.dirname(os.path.abspath(args.sheet)) or "."
    os.makedirs(d, exist_ok=True)

    cols = args.cols
    rows = (len(files) + cols - 1) // cols
    cw, ch = args.cell_width, args.cell_height
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    report = []

    for i, f in enumerate(files):
        with Image.open(f) as im:
            box = find_page(im, args.thresh)
            crop = im.crop(box)
            report.append({
                "index": i,
                "file": f,
                "size": list(im.size),
                "crop_box": list(box),
                "cropped_size": [box[2] - box[0], box[3] - box[1]],
                "cropped_ratio": round((box[3] - box[1]) / max(1, box[2] - box[0]), 3),
                "kept_fraction": round(((box[2] - box[0]) * (box[3] - box[1]))
                                       / (im.width * im.height), 3),
            })
            sheet.paste(crop.convert("RGB").resize((cw, ch), Image.LANCZOS),
                        ((i % cols) * cw, (i // cols) * ch))

    sheet.save(args.sheet)

    # 页脚拼贴：按固定页数分批，避免 50 页以上拼成一张超长图。
    strip_paths = []
    if not args.no_footer:
        for chunk_start in range(0, len(files), FOOTER_STRIP_PAGES):
            chunk = files[chunk_start:chunk_start + FOOTER_STRIP_PAGES]
            crops = []
            for f, r in zip(chunk, report[chunk_start:chunk_start + FOOTER_STRIP_PAGES]):
                with Image.open(f) as im:
                    page_im = im.crop(tuple(r["crop_box"])).convert("RGB")
                fh = max(40, int(page_im.height * args.footer_frac))
                c = page_im.crop((0, page_im.height - fh, page_im.width, page_im.height))
                if args.footer_zoom != 1.0:
                    c = c.resize((round(c.width * args.footer_zoom),
                                  round(c.height * args.footer_zoom)), Image.LANCZOS)
                crops.append(c)
            sw = max(c.width for c in crops)
            gap = 6
            strip = Image.new("RGB", (sw, sum(c.height for c in crops) + gap * len(crops)), "gray")
            y = 0
            for c in crops:
                strip.paste(c, (0, y))
                y += c.height + gap
            strip_path = (os.path.splitext(args.sheet)[0]
                          + f"_footers_{chunk_start // FOOTER_STRIP_PAGES + 1:03d}.png")
            strip.save(strip_path)
            strip_paths.append(strip_path)

    print(json.dumps({"count": len(files), "sheet": args.sheet,
                      "footer_strips": strip_paths, "pages": report},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- build


def cmd_build(args):
    files = collect(args.inputs, dedupe=args.dedupe)
    if not files:
        sys.exit("没有找到图片")
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    jpg_paths = []
    warn = []

    for i, f in enumerate(files, 1):
        with Image.open(f) as opened:
            im = opened.convert("RGB")
            before = im.size
            if not args.no_crop:
                im = im.crop(find_page(im, args.thresh, args.margin))
            # 黑边自检在加 A4 白边之前做，否则白画布边缘会掩盖黑边残留。
            edge_check_im = im
            im = enhance(im, args.target_width, args.sharpen, args.contrast,
                         args.radius, args.threshold, args.grayscale,
                         args.whitepoint, args.fit_a4, args.page_margin,
                         args.ink_save, args.ink_limit)
            name = f"{i:02d}.jpg"
            jpg_path = os.path.join(args.out, name)
            im.save(jpg_path, "JPEG", quality=args.quality,
                    dpi=(args.dpi, args.dpi), optimize=True)
            edges, min_edge = edge_warnings(edge_check_im, args.edge_min)
            if min_edge < args.edge_min:
                warn.append({"page": i, "edge_means": edges})
            manifest.append({"page": i, "source": os.path.basename(f),
                             "before": list(before), "after": list(im.size),
                             "out": name, "edge_means": edges})
            jpg_paths.append(jpg_path)

    pdf = None
    if not args.no_pdf:
        stem = args.pdf_name or os.path.basename(args.out.rstrip("/"))
        pdf = os.path.join(args.out, stem + ".pdf")
        # 不把所有页面像素同时放在内存：JPG 已在磁盘，PDF 编码时逐页读取。
        with Image.open(jpg_paths[0]) as first:
            # append_images 用 ImageFile 对象列表；它们按需解码，避免 OOM。
            rest = [Image.open(p) for p in jpg_paths[1:]]
            try:
                first.save(pdf, "PDF", resolution=float(args.dpi),
                           save_all=True, append_images=rest)
            finally:
                for im in rest:
                    im.close()

    print(json.dumps({"out": args.out, "pages": len(manifest), "pdf": pdf,
                      "manifest": manifest, "dark_edge_warnings": warn},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- verify


def cmd_verify(args):
    """对 build 输出目录做快速视觉自检。"""
    require_deps()
    sheet_path = args.sheet
    d = os.path.dirname(os.path.abspath(sheet_path)) or "."
    os.makedirs(d, exist_ok=True)

    # 收集每个输出目录里的最终 JPG（排除临时/子目录里的散图）。
    dirs = []
    for p in args.dirs:
        if os.path.isdir(p):
            jpgs = sorted(
                [os.path.join(p, f) for f in os.listdir(p)
                 if f.lower().endswith(EXTS) and not f.startswith("._")],
                key=natural_key
            )
            if jpgs:
                dirs.append({"dir": p, "jpgs": jpgs})
        else:
            sys.exit(f"不是目录: {p}")

    if not dirs:
        sys.exit("没有找到可验证的 JPG")

    total = sum(len(x["jpgs"]) for x in dirs)
    cols = args.cols
    rows = (total + cols - 1) // cols
    cw, ch = args.cell_width, args.cell_height
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    pos = 0
    stats = []
    warn = []

    for group in dirs:
        group_sizes = []
        for f in group["jpgs"]:
            with Image.open(f) as im:
                im = im.convert("RGB")
                edges, min_edge = edge_warnings(im, args.edge_min)
                if min_edge < args.edge_min:
                    warn.append({"dir": group["dir"], "file": f,
                                 "edge_means": edges})
                group_sizes.append(list(im.size))
                sheet.paste(im.resize((cw, ch), Image.LANCZOS),
                            ((pos % cols) * cw, (pos // cols) * ch))
                pos += 1
        stats.append({
            "dir": group["dir"],
            "pages": len(group["jpgs"]),
            "sizes": group_sizes,
            "unique_sizes": sorted(set(map(tuple, group_sizes))),
        })

    sheet.save(sheet_path)
    print(json.dumps({"sheet": sheet_path, "total_pages": total,
                      "stats": stats, "edge_warnings": warn},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- split


def cmd_split(args):
    """横屏双页图：按中缝切成左右两页，输出为 *_L.jpg / *_R.jpg。"""
    files = collect(args.inputs, dedupe=False)
    if not files:
        sys.exit("没有找到图片")
    os.makedirs(args.out, exist_ok=True)
    made = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(f)))
        if parent and parent not in (".", os.path.sep):
            stem = f"{parent}_{stem}"
        with Image.open(f) as opened:
            im = opened.convert("RGB")
            w, h = im.size
            left = im.crop((0, 0, w // 2, h))
            right = im.crop((w // 2, 0, w, h))
            lp = os.path.join(args.out, f"{stem}_L.jpg")
            rp = os.path.join(args.out, f"{stem}_R.jpg")
            left.save(lp, "JPEG", quality=args.quality)
            right.save(rp, "JPEG", quality=args.quality)
            made.extend([lp, rp])
    print(json.dumps({"out": args.out, "created": made},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- pack


def cmd_pack(args):
    d = os.path.dirname(os.path.abspath(args.zip)) or "."
    os.makedirs(d, exist_ok=True)
    n = 0
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in args.dirs:
            base = os.path.basename(folder.rstrip("/"))
            for root, _, files in os.walk(folder):
                for f in sorted(files, key=natural_key):
                    full = os.path.join(root, f)
                    z.write(full, os.path.join(base, os.path.relpath(full, folder)))
                    n += 1
    print(json.dumps({"zip": args.zip, "files": n,
                      "size_mb": round(os.path.getsize(args.zip) / 1e6, 1)},
                     ensure_ascii=False))


# ---------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser(description="试卷/讲义截图 → 打印稿")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("survey", help="生成缩略图联表 + 分页页脚拼贴 + 裁剪诊断")
    s.add_argument("inputs", nargs="+", help="图片/目录/zip")
    s.add_argument("--sheet", default="sheet.png")
    s.add_argument("--cols", type=int, default=6)
    s.add_argument("--cell-width", type=int, default=420)
    s.add_argument("--cell-height", type=int, default=594)
    s.add_argument("--thresh", type=int, default=150)
    s.add_argument("--footer-frac", type=float, default=0.12,
                   help="页脚拼贴取纸面底部多少比例，默认 12%%")
    s.add_argument("--footer-zoom", type=float, default=1.5)
    s.add_argument("--no-footer", action="store_true")
    s.add_argument("--dedupe", action="store_true",
                   help="先按 MD5 去掉完全相同的图片")
    s.set_defaults(func=cmd_survey)

    b = sub.add_parser("build", help="裁边+锐化+出 JPG 和 PDF")
    b.add_argument("inputs", nargs="+")
    b.add_argument("--out", required=True, help="输出目录，目录名会成为 PDF 名")
    b.add_argument("--pdf-name", default=None)
    b.add_argument("--target-width", type=int, default=2480,
                   help="目标像素宽，2480≈A4@300dpi；开启 A4 画布时是画布宽度")
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--sharpen", type=int, default=130, help="USM 强度百分比，0=不锐化")
    b.add_argument("--radius", type=float, default=3.0)
    b.add_argument("--threshold", type=int, default=2)
    b.add_argument("--contrast", type=float, default=1.18)
    b.add_argument("--whitepoint", type=int, default=255,
                   help="<255 时把亮度不低于该值的像素推成纯白，治灰底")
    b.add_argument("--grayscale", action="store_true", help="转灰度，黑白打印更省墨")
    b.add_argument("--ink-save", action="store_true",
                   help="省墨模式：调淡大面积深色区域，尽量保留文字黑色")
    b.add_argument("--ink-limit", type=float, default=0.45,
                   help="省墨模式保留的墨量比例，默认 0.45（即大面积深色调到约 45%% 墨量）")
    b.add_argument("--quality", type=int, default=95)
    b.add_argument("--margin", type=int, default=0, help="裁剪框再向内收 N 像素")
    b.add_argument("--thresh", type=int, default=150)
    b.add_argument("--edge-min", type=int, default=200, help="四边平均亮度低于此值就报警")
    b.add_argument("--no-crop", action="store_true")
    b.add_argument("--no-pdf", action="store_true")
    b.add_argument("--fit-a4", dest="fit_a4", action="store_true", default=True,
                   help="输出到标准 A4 白底画布（默认开启）")
    b.add_argument("--no-fit-a4", dest="fit_a4", action="store_false",
                   help="关闭 A4 画布，只按 target-width 缩放宽度")
    b.add_argument("--page-margin", type=int, default=40,
                   help="A4 画布四周留白像素，默认 40")
    b.add_argument("--dedupe", action="store_true",
                   help="先按 MD5 去掉完全相同的图片")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="对 build 输出做视觉自检")
    v.add_argument("dirs", nargs="+", help="build 的输出目录")
    v.add_argument("--sheet", default="verify_sheet.png")
    v.add_argument("--cols", type=int, default=4)
    v.add_argument("--cell-width", type=int, default=400)
    v.add_argument("--cell-height", type=int, default=566)
    v.add_argument("--edge-min", type=int, default=200)
    v.set_defaults(func=cmd_verify)

    sp = sub.add_parser("split", help="横屏双页图按中缝切分为两页")
    sp.add_argument("inputs", nargs="+", help="图片/目录/zip")
    sp.add_argument("--out", required=True, help="输出目录")
    sp.add_argument("--quality", type=int, default=95)
    sp.set_defaults(func=cmd_split)

    p = sub.add_parser("pack", help="打包成 zip")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--zip", required=True)
    p.set_defaults(func=cmd_pack)

    args = ap.parse_args()
    require_deps()
    args.func(args)


if __name__ == "__main__":
    main()
