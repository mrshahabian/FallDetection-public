"""
Generate pipeline diagrams for the academic report.
Creates visual pipeline diagrams for each method.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("evaluation/report_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_zeroshot_pipeline():
    """Create zero-shot VLM pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Zero-Shot VLM Pipeline', ha='center', fontsize=18, fontweight='bold')
    
    # Video input
    video_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(video_box)
    ax.text(1.25, 8, 'Video\nInput', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Frame sampling
    frame_box = FancyBboxPatch((2.5, 7.5), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(frame_box)
    ax.text(3.25, 8, 'Frame\nSampling\n(1 or 8)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # CLIP Vision Encoder
    vision_box = FancyBboxPatch((4.5, 7.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(vision_box)
    ax.text(5.25, 8, 'CLIP Vision\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Image embeddings
    img_emb_box = FancyBboxPatch((7, 7.5), 1.5, 1, boxstyle="round,pad=0.1",
                                facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(img_emb_box)
    ax.text(7.75, 8, 'Image\nEmbeddings\n[N, 512]', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Text prompts
    prompt_box = FancyBboxPatch((0.5, 5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(1.25, 5.5, 'Text\nPrompts\n(198)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # CLIP Text Encoder
    text_box = FancyBboxPatch((2.5, 5), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(text_box)
    ax.text(3.25, 5.5, 'CLIP Text\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Text embeddings
    text_emb_box = FancyBboxPatch((4.5, 5), 1.5, 1, boxstyle="round,pad=0.1",
                                  facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(text_emb_box)
    ax.text(5.25, 5.5, 'Text\nEmbeddings\n[198, 512]', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Similarity computation
    sim_box = FancyBboxPatch((7, 5), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(sim_box)
    ax.text(7.75, 5.5, 'Cosine\nSimilarity\n[N, 198]', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Aggregate
    agg_box = FancyBboxPatch((2.5, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#16a085', edgecolor='black', linewidth=2)
    ax.add_patch(agg_box)
    ax.text(3.25, 3, 'Aggregate\nAcross\nFrames', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Contrastive probability
    prob_box = FancyBboxPatch((4.5, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(prob_box)
    ax.text(5.25, 3, 'Contrastive\nProbability\nCalculation', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output
    out_box = FancyBboxPatch((7, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(out_box)
    ax.text(7.75, 3, 'Fall\nProbability', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2, 8), (2.5, 8)),  # Video to frame sampling
        ((4, 8), (4.5, 8)),  # Frame to vision encoder
        ((6, 8), (7, 8)),    # Vision to embeddings
        ((2, 5.5), (2.5, 5.5)),  # Prompts to text encoder
        ((4, 5.5), (4.5, 5.5)),  # Text encoder to embeddings
        ((6, 5.5), (7, 5.5)),    # Text embeddings to similarity
        ((7.75, 7), (3.25, 6.5)),  # Similarity to aggregate
        ((4, 3), (4.5, 3)),  # Aggregate to probability
        ((6, 3), (7, 3)),    # Probability to output
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=0.1" if abs(y1-y2) > 0.5 else None)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zeroshot_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'zeroshot_pipeline.png'}")

def create_fewshot_pipeline():
    """Create few-shot VLM pipeline diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Training pipeline
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.text(5, 9.5, 'Few-Shot VLM Training Pipeline', ha='center', fontsize=16, fontweight='bold')
    
    # Training videos
    train_box = FancyBboxPatch((0.5, 7), 2, 1, boxstyle="round,pad=0.1",
                              facecolor='#3498db', edgecolor='black', linewidth=2)
    ax1.add_patch(train_box)
    ax1.text(1.5, 7.5, 'Training Videos\n(32 videos)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Frame sampling
    frame_box = FancyBboxPatch((3, 7), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax1.add_patch(frame_box)
    ax1.text(3.75, 7.5, 'Frame\nSampling', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # CLIP (frozen)
    clip_box = FancyBboxPatch((5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax1.add_patch(clip_box)
    ax1.text(5.75, 7.5, 'CLIP Vision\nEncoder\n(FROZEN)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Embeddings
    emb_box = FancyBboxPatch((7, 7), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax1.add_patch(emb_box)
    ax1.text(7.75, 7.5, 'Embeddings\n[512]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Linear classifier
    classifier_box = FancyBboxPatch((2, 4.5), 2, 1, boxstyle="round,pad=0.1",
                                   facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax1.add_patch(classifier_box)
    ax1.text(3, 5, 'Linear Classifier\n(512 → 1)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Training
    train_proc_box = FancyBboxPatch((5, 4.5), 2, 1, boxstyle="round,pad=0.1",
                                   facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax1.add_patch(train_proc_box)
    ax1.text(6, 5, 'Training\n(BCE Loss)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Trained classifier
    trained_box = FancyBboxPatch((2, 2), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax1.add_patch(trained_box)
    ax1.text(3, 2.5, 'Trained\nClassifier', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows1 = [
        ((2.5, 7.5), (3, 7.5)),
        ((4.5, 7.5), (5, 7.5)),
        ((6.5, 7.5), (7, 7.5)),
        ((7.75, 6.5), (3, 5)),
        ((4, 5), (5, 5)),
        ((7, 5), (3, 3)),
    ]
    
    for (x1, y1), (x2, y2) in arrows1:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=0.1" if abs(y1-y2) > 0.5 else None)
        ax1.add_patch(arrow)
    
    # Inference pipeline
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.text(5, 9.5, 'Few-Shot VLM Inference Pipeline', ha='center', fontsize=16, fontweight='bold')
    
    # Test video
    test_box = FancyBboxPatch((0.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#3498db', edgecolor='black', linewidth=2)
    ax2.add_patch(test_box)
    ax2.text(1.25, 7.5, 'Test\nVideo', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Frame sampling
    frame2_box = FancyBboxPatch((2.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax2.add_patch(frame2_box)
    ax2.text(3.25, 7.5, 'Frame\nSampling', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # CLIP (frozen)
    clip2_box = FancyBboxPatch((4.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax2.add_patch(clip2_box)
    ax2.text(5.25, 7.5, 'CLIP Vision\nEncoder\n(FROZEN)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Embedding
    emb2_box = FancyBboxPatch((6.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax2.add_patch(emb2_box)
    ax2.text(7.25, 7.5, 'Embedding\n[512]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Trained classifier
    classifier2_box = FancyBboxPatch((2.5, 4.5), 2, 1, boxstyle="round,pad=0.1",
                                    facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax2.add_patch(classifier2_box)
    ax2.text(3.5, 5, 'Trained\nClassifier', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    out2_box = FancyBboxPatch((5.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax2.add_patch(out2_box)
    ax2.text(6.25, 5, 'Fall\nProbability', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows2 = [
        ((2, 7.5), (2.5, 7.5)),
        ((4, 7.5), (4.5, 7.5)),
        ((6, 7.5), (6.5, 7.5)),
        ((7.25, 6.5), (3.5, 5)),
        ((4.5, 5), (5.5, 5)),
    ]
    
    for (x1, y1), (x2, y2) in arrows2:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=0.1" if abs(y1-y2) > 0.5 else None)
        ax2.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fewshot_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'fewshot_pipeline.png'}")

def create_supervised_pipeline():
    """Create supervised method pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.5, 'Supervised Method Pipeline (2D CNN, 3D CNN, ViT)', ha='center', fontsize=18, fontweight='bold')
    
    # Video input
    video_box = FancyBboxPatch((0.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(video_box)
    ax.text(1.25, 7.5, 'Video\nInput', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # YOLO skeleton extraction
    yolo_box = FancyBboxPatch((2.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(yolo_box)
    ax.text(3.25, 7.5, 'YOLOv11-pose\nSkeleton\nExtraction', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Keypoints
    keypoint_box = FancyBboxPatch((4.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                                 facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(keypoint_box)
    ax.text(5.25, 7.5, 'Keypoints\n[17 joints,\n2 coords]', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Normalization
    norm_box = FancyBboxPatch((6.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(norm_box)
    ax.text(7.25, 7.5, 'Normalization\n(center-on-hip,\nscale-to-[0,1])', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Window extraction
    window_box = FancyBboxPatch((1, 4.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(window_box)
    ax.text(1.75, 5, 'Window\nExtraction\n(32 frames)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Format conversion
    format_box = FancyBboxPatch((3, 4.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(format_box)
    ax.text(3.75, 5, 'Format\nConversion\n(2D/3D)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Model (2D CNN / 3D CNN / ViT)
    model_box = FancyBboxPatch((5, 4.5), 2, 1, boxstyle="round,pad=0.1",
                              facecolor='#16a085', edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(6, 5, 'Deep Learning Model\n(2D CNN / 3D CNN / ViT)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    out_box = FancyBboxPatch((7.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1",
                            facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(out_box)
    ax.text(8.25, 5, 'Binary\nClassification\n(Fall/Non-Fall)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        ((2, 7.5), (2.5, 7.5)),
        ((4, 7.5), (4.5, 7.5)),
        ((6, 7.5), (6.5, 7.5)),
        ((7.25, 6.5), (1.75, 5.5)),
        ((3.25, 5), (3, 5)),
        ((4.5, 5), (5, 5)),
        ((7, 5), (7.5, 5)),
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=0.1" if abs(y1-y2) > 0.5 else None)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supervised_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'supervised_pipeline.png'}")

def main():
    """Generate all pipeline diagrams."""
    print("Generating pipeline diagrams...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    create_zeroshot_pipeline()
    create_fewshot_pipeline()
    create_supervised_pipeline()
    
    print("\n✓ All pipeline diagrams generated successfully!")
    print(f"Figures saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


