import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def initial_superpixels(img, grid=16):
    """
    img: [3, H, W] float
    grid: grid size for superpixels
    """
    C, H, W = img.shape
    labels = torch.zeros((H, W), dtype=torch.long)

    region_id = 0
    for y in range(0, H, grid):
        for x in range(0, W, grid):
            labels[y:y+grid, x:x+grid] = region_id
            region_id += 1
    return labels, region_id



def compute_region_props(img, labels, num_regions, bins=16):
    """
    为每个 region 计算:
    - color histogram
    - bounding box
    """
    C, H, W = img.shape
    props = []

    for r in range(num_regions):
        ys, xs = (labels == r).nonzero(as_tuple=True)
        if len(xs) == 0:
            props.append(None)
            continue

        # bounding box
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()

        region_pixels = img[:, ys, xs]   # [3, N]

        # color hist
        hist = []
        for c in range(3):
            h = torch.histc(region_pixels[c], bins=bins, min=0.0, max=1.0)
            hist.append(h / h.sum())
        hist = torch.cat(hist)   # [bins*3]

        props.append({
            "id": r,
            "hist": hist,
            "bbox": [x1, y1, x2, y2],
            "pixels": len(xs)
        })

    return props


def similarity(r1, r2):
    if r1 is None or r2 is None:
        return -1

    # color similarity = histogram intersection
    col_sim = torch.sum(torch.min(r1["hist"], r2["hist"])).item()

    # size similarity
    size_sim = 1.0 / (r1["pixels"] + r2["pixels"])

    return col_sim + size_sim



def merge_regions(props, labels):
    H, W = labels.shape

    proposals = []

    while True:
        # 找最相似的一对区域
        best_sim = -1
        best_pair = None

        for i in range(len(props)):
            for j in range(i+1, len(props)):
                if props[i] is None or props[j] is None:
                    continue
                sim = similarity(props[i], props[j])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)

        if best_pair is None or best_sim < 0:
            break

        a, b = best_pair

        # 合并 labels
        labels[labels == b] = a

        # 重新计算 region a
        ys, xs = (labels == a).nonzero(as_tuple=True)
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()

        props[a]["bbox"] = [x1, y1, x2, y2]
        props[a]["pixels"] = len(xs)

        # 删除 region b
        props[b] = None

        # ✨ 记录 proposal（关键）
        proposals.append([x1, y1, x2-x1, y2-y1])

    return proposals


def run_selective_search(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2,0,1).float()  # [3,H,W]

    labels, n_regions = initial_superpixels(img)

    props = compute_region_props(img, labels, n_regions)

    proposals = merge_regions(props, labels)

    print("生成候选框数量:", len(proposals))
    return img.permute(1,2,0).numpy(), proposals


def show(img, proposals, N=50):
    plt.figure(figsize=(8,8))
    plt.imshow(img)

    for i, (x,y,w,h) in enumerate(proposals[:N]):
        rect = plt.Rectangle((x,y), w,h, fill=False, color='red', linewidth=1)
        plt.gca().add_patch(rect)

    plt.show()


if __name__ == "__main__":
    img, proposals = run_selective_search("test.jpg")
    show(img, proposals, N=50)




