#!/usr/bin/env python3
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm

import chromadb
import open_clip
import warnings

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


# available models
#for m, p in open_clip.list_pretrained():
#    if "ViT-L-14" in m:
#        print(m, "->", p)


warnings.filterwarnings(
    "ignore",
    message="QuickGELU mismatch.*"
)



IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_images(root: Path, recursive: bool = True) -> List[Path]:
    out = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def load_rgb(path: Path) -> Image.Image:
    # Robust loading
    img = Image.open(path)
    # Some images have strange modes (P, RGBA, CMYK...): convert to RGB
    return img.convert("RGB")


@torch.inference_mode()
def embed_batch(
    model,
    preprocess,
    device: str,
    paths: List[Path],
    use_fp16: bool = True,
    base_dir: Path = None,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    imgs = []
    metas = []

    for p in paths:
        try:
            img = load_rgb(p)
            w, h = img.size
            t = preprocess(img)  # resize+center-crop+normalize (standard OpenCLIP)

            #print(t.shape)   # should be [3, 336, 336]

            imgs.append(t)
            st = p.stat()
            # Use relative path if base_dir is provided
            path_str = str(p)
            if base_dir:
                try:
                    rel_path = p.relative_to(base_dir)
                    path_str = str(rel_path)
                except ValueError:
                    # Path is not relative to base_dir, keep absolute
                    pass
            metas.append(
                {
                    "path": path_str,
                    "w": int(w),
                    "h": int(h),
                    "mtime": int(st.st_mtime),
                }
            )
        except Exception as e:
            metas.append({"path": str(p), "error": str(e)})
            # placeholder: skip the image
            continue

    if not imgs:
        return [], []

    batch = torch.stack(imgs, dim=0).to(device)

    if use_fp16 and device.startswith("cuda"):
        batch = batch.half()
        model = model.half()

    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalization
    feats = feats.detach().float().cpu().numpy()

    # Convert to Python lists for Chroma
    embeddings = feats.tolist()

    # Note: metas contains possible errors; here we return only the ok ones
    ok_metas = [m for m in metas if "error" not in m]
    return embeddings, ok_metas


def main(
    images_dir: str,
    chroma_dir: str = "./db/chroma",
    collection: str = "images_openclip_vitl14_336",
    batch_size: int = 512,
    device: str = "cuda",
    model_name: str = "ViT-L-14-336",
    pretrained: str = "openai",
    recursive: bool = True,
):
    root = Path(images_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: directory not found: {root}")
        sys.exit(1)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available: switching to CPU.")
        device = "cpu"

    print(f"[INFO] Scanning images in: {root}")
    paths = list_images(root, recursive=recursive)
    print(f"[INFO] Found {len(paths)} images")

    # Standard model + preprocess
    print(f"[INFO] Loading OpenCLIP {model_name} ({pretrained}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, force_quick_gelu=True
    )
    print(preprocess)
    model = model.to(device)
    model.eval()

    # Persistent Chroma
    chroma_path = Path(chroma_dir).expanduser().resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},  # consistent with normalized embeddings
    )

    # To avoid re-insertions: load already present IDs (for 100k it's fine)
    # If it becomes heavy, we optimize in chunks.
    existing = set()
    try:
        # Chroma doesn't expose a perfect "list all ids" for huge collections in all versions,
        # so we make an attempt: if it fails, it will continue anyway.
        got = col.get(include=[])
        for _id in got.get("ids", []):
            existing.add(_id)
        print(f"[INFO] Already present IDs (estimate): {len(existing)}")
    except Exception:
        print("[WARN] Cannot read existing IDs: continuing without ID cache.")
        existing = set()

    added = 0
    skipped = 0
    duplicates = 0
    errors = 0

    t0 = time.time()
    batch: List[Path] = []

    def flush(batch_paths: List[Path]):
        nonlocal added, skipped, duplicates, errors

        # Calculate id (sha1) and skip those already present
        ids = []
        ok_paths = []
        metas_extra = []

        for p in batch_paths:
            try:
                sid = sha1_file(p)
                if sid in existing:
                    skipped += 1
                    continue
                ids.append(sid)
                ok_paths.append(p)
            except Exception as e:
                errors += 1
                continue

        if not ok_paths:
            return

        embeddings, metas = embed_batch(
            model=model,
            preprocess=preprocess,
            device=device,
            paths=ok_paths,
            use_fp16=True,
            base_dir=root,
        )

        # metas must be aligned with embeddings and ids
        # embed_batch returns only OK images; so we must rebuild consistent ids/metas:
        # use path to map (must match the format used in embed_batch)
        path_to_id = {}
        for p, sid in zip(ok_paths, ids):
            try:
                rel_path = p.relative_to(root)
                path_to_id[str(rel_path)] = sid
            except ValueError:
                path_to_id[str(p)] = sid
        
        ids_ok = []
        metas_ok = []
        seen_in_batch = set()
        for m in metas:
            pid = path_to_id.get(m["path"])
            if pid is None:
                errors += 1
                continue
            # Skip duplicates within this batch
            if pid in seen_in_batch:
                duplicates += 1
                continue
            seen_in_batch.add(pid)
            # Add sha1 in metadata (useful)
            m["sha1"] = pid
            ids_ok.append(pid)
            metas_ok.append(m)

        if not ids_ok:
            return

        col.add(ids=ids_ok, embeddings=embeddings[: len(ids_ok)], metadatas=metas_ok)
        for pid in ids_ok:
            existing.add(pid)
        added += len(ids_ok)

    for p in tqdm(paths, desc="Embedding"):
        batch.append(p)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    if batch:
        flush(batch)

    dt = time.time() - t0
    print("\n[DONE]")
    print(f"  Added      : {added}")
    print(f"  Skipped    : {skipped} (already present)")
    print(f"  Duplicates : {duplicates} (within batch)")
    print(f"  Errors     : {errors}")
    print(f"  Time       : {dt:.1f}s")
    print(f"  Chroma     : {chroma_path}")
    print(f"  Collection : {collection}")




