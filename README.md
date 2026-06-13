# Ai-Smart-Shelf-Vision — Automated Retail Product & Brand Recognition

## Overview

ShelfSense is an end-to-end computer vision pipeline that automatically
detects, segments, and identifies retail products (with brand names) on
store shelf images. It combines **SAM3** for universal object segmentation
with a **Siamese Network (SA-Siam)** for product identification via
similarity matching.

The output is the original shelf image, with each detected product
outlined by its segmentation mask and labeled with its predicted
product name and brand.

---

## Pipeline Architecture
markdown# ShelfSense — Automated Retail Product & Brand Recognition

SAM3 + SA-Siam pipeline for automatic retail product detection, segmentation, and brand identification on shelf images.

## Overview

ShelfSense is an end-to-end computer vision pipeline that automatically detects, segments, and identifies retail products (with brand names) on store shelf images. It combines **SAM3** (Segment Anything Model 3) for universal object segmentation with a **Siamese Network (SA-Siam)** for product identification via similarity matching.

The output is the original shelf image, with each detected product outlined by its segmentation mask and labeled with its predicted product name and brand.

## Pipeline Architecture
Input Shelf Image (X)

│

▼

┌─────────────────┐

│   SAM3 (hψ)      │   → Segments all objects

└─────────────────┘

│

▼

Masks M(i) = [{segmentation, bbox, ...}, ...]

│

▼

┌─────────────────┐

│   crop(·)        │   → Cuts out each object using its mask

└─────────────────┘

│

▼

Cropped Images X(i,1)c ... X(i,n)c

│

▼

┌─────────────────┐

│  SA-Siam Network │   → fθ (feature extractor) + gϕ (comparator)

│  (gϕ ∘ fθ)        │

└─────────────────┘

│

▼

Predictions ŷ(i,1) ... ŷ(i,n)

→ Product Name + Brand Name per object

│

▼

Annotated Output Image

(mask outline + product/brand label per object)

## How It Works

### 1. Segmentation (SAM3)
The input shelf image is passed to SAM3, which detects every object on the shelf and returns a list of masks. Each mask is a binary `(H, W)` array marking the exact pixels belonging to one product.

### 2. Cropping
For each mask, the corresponding region is cropped from the original image, producing a clean, isolated image of a single product.

### 3. Feature Extraction & Identification (SA-Siam)
Each cropped product image is passed through the Siamese Network.
- `fθ`: extracts a feature embedding (visual fingerprint) of the cropped product.
- `gϕ`: compares this embedding against a reference database of known product embeddings and returns the closest match — the predicted **product name and brand**.

### 4. Annotation & Output
The original image is rendered again with:
- The segmentation mask outline drawn around each detected product
- A text label above each object showing its predicted product name and brand

## Example

**Input:** A photo of a cereal shelf containing multiple boxes.

**Output:** The same photo, where:
- Each cereal box has a colored outline around it (from its SAM3 mask)
- Above each box is a label such as:
  - `Kellogg's — Coco Pops`
  - `Nestlé — Cheerios`
  - `Quaker — Oat Squares`

## Tech Stack

| Component | Tool / Model |
|---|---|
| Segmentation | SAM3 (Segment Anything Model 3) |
| Feature Extraction | SA-Siam / ViT backbone |
| Comparison Head | Siamese metric-learning head (gϕ) |
| Dataset | Custom retail product dataset (positive/negative pairs) |
