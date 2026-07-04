"""
Generate visualizations for the academic report.
Creates graphs, charts, and diagrams for the comprehensive fall detection report.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Results directory
RESULTS_DIR = Path("evaluation/results/complete_evaluation_20260104_210115")
OUTPUT_DIR = Path("evaluation/report_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_results():
    """Load all result files."""
    results = {}
    
    # Supervised methods
    with open(RESULTS_DIR / "supervised_single_person/detailed_results.json") as f:
        results['supervised'] = json.load(f)
    
    # VLM 1 frame
    with open(RESULTS_DIR / "vlm_1frames/detailed_results.json") as f:
        results['vlm_1frame'] = json.load(f)
    
    # VLM 8 frames
    with open(RESULTS_DIR / "vlm_8frames/detailed_results.json") as f:
        results['vlm_8frames'] = json.load(f)
    
    return results

def create_comparison_bar_chart(results):
    """Create bar chart comparing all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    methods = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    inference_time = []
    
    # Supervised methods
    for method in results['supervised']:
        name = method['method_name'].replace('_', ' ').title()
        methods.append(name)
        accuracy.append(method['metrics']['accuracy'])
        precision.append(method['metrics']['precision'])
        recall.append(method['metrics']['recall'])
        f1_score.append(method['metrics']['f1_score'])
        inference_time.append(method['metrics']['avg_inference_time'])
    
    # VLM methods (use 1 frame for inference time)
    for method in results['vlm_1frame']:
        name = method['method_name'].split('(')[0].strip()
        if name not in methods:
            methods.append(name)
            accuracy.append(method['metrics']['accuracy'])
            precision.append(method['metrics']['precision'])
            recall.append(method['metrics']['recall'])
            f1_score.append(method['metrics']['f1_score'])
            inference_time.append(method['metrics']['avg_inference_time'])
    
    x = np.arange(len(methods))
    width = 0.2
    
    # Accuracy
    axes[0, 0].bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2ecc71')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].legend()
    
    # Precision, Recall, F1
    axes[0, 1].bar(x - width, precision, width, label='Precision', color='#3498db')
    axes[0, 1].bar(x, recall, width, label='Recall', color='#e74c3c')
    axes[0, 1].bar(x + width, f1_score, width, label='F1-Score', color='#f39c12')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision, Recall, F1-Score Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].legend()
    
    # Inference Time
    axes[1, 0].bar(x, inference_time, width*2, color='#9b59b6')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Average Inference Time per Window')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # ROC AUC and PR AUC
    roc_auc = []
    pr_auc = []
    for method in results['supervised']:
        roc_auc.append(method['metrics']['roc_auc'])
        pr_auc.append(method['metrics']['pr_auc'])
    for method in results['vlm_1frame']:
        roc_auc.append(method['metrics']['roc_auc'])
        pr_auc.append(method['metrics']['pr_auc'])
    
    axes[1, 1].bar(x - width/2, roc_auc, width, label='ROC AUC', color='#1abc9c')
    axes[1, 1].bar(x + width/2, pr_auc, width, label='PR AUC', color='#e67e22')
    axes[1, 1].set_ylabel('AUC Score')
    axes[1, 1].set_title('ROC AUC and PR AUC Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'comparison_bar_chart.png'}")

def create_confusion_matrices(results):
    """Create confusion matrix heatmaps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Simple Fall Detection
    cm_simple = np.array([[19, 0], [16, 5]])
    sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[0].set_title('Simple Fall Detection\nAccuracy: 60.0%', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=12)
    
    # Zero-Shot VLM
    cm_zeroshot = np.array([[18, 1], [0, 21]])
    sns.heatmap(cm_zeroshot, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[1].set_title('Zero-Shot VLM\nAccuracy: 97.5%', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=12)
    
    # Few-Shot VLM
    cm_fewshot = np.array([[19, 0], [0, 21]])
    sns.heatmap(cm_fewshot, annot=True, fmt='d', cmap='Greens', ax=axes[2],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[2].set_title('Few-Shot VLM\nAccuracy: 100.0%', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Actual', fontsize=12)
    axes[2].set_xlabel('Predicted', fontsize=12)
    
    # 2D CNN
    cm_2dcnn = np.array([[19, 0], [0, 21]])
    sns.heatmap(cm_2dcnn, annot=True, fmt='d', cmap='Oranges', ax=axes[3],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[3].set_title('2D CNN (ResNet)\nAccuracy: 100.0%', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Actual', fontsize=12)
    axes[3].set_xlabel('Predicted', fontsize=12)
    
    # 3D CNN
    cm_3dcnn = np.array([[19, 0], [0, 21]])
    sns.heatmap(cm_3dcnn, annot=True, fmt='d', cmap='Oranges', ax=axes[4],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[4].set_title('3D CNN (Simple)\nAccuracy: 100.0%', fontsize=14, fontweight='bold')
    axes[4].set_ylabel('Actual', fontsize=12)
    axes[4].set_xlabel('Predicted', fontsize=12)
    
    # ViT
    cm_vit = np.array([[19, 0], [0, 21]])
    sns.heatmap(cm_vit, annot=True, fmt='d', cmap='Oranges', ax=axes[5],
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    axes[5].set_title('Vision Transformer (ViT)\nAccuracy: 100.0%', fontsize=14, fontweight='bold')
    axes[5].set_ylabel('Actual', fontsize=12)
    axes[5].set_xlabel('Predicted', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")

def create_inference_time_comparison(results):
    """Create inference time comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = []
    times = []
    colors = []
    
    # Supervised methods
    for method in results['supervised']:
        name = method['method_name'].replace('_', ' ').title()
        methods.append(name)
        times.append(method['metrics']['avg_inference_time'])
        colors.append('#e74c3c')  # Red for supervised
    
    # VLM methods
    for method in results['vlm_1frame']:
        name = method['method_name'].split('(')[0].strip()
        if name not in methods:
            methods.append(name)
            times.append(method['metrics']['avg_inference_time'])
            colors.append('#2ecc71')  # Green for VLM
    
    # Sort by time
    sorted_data = sorted(zip(methods, times, colors), key=lambda x: x[1])
    methods, times, colors = zip(*sorted_data)
    
    bars = ax.barh(methods, times, color=colors, alpha=0.8)
    ax.set_xlabel('Average Inference Time per Window (seconds)', fontsize=14)
    ax.set_title('Inference Time Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(time + 0.005, i, f'{time:.4f}s', va='center', fontsize=10)
    
    # Add legend
    red_patch = mpatches.Patch(color='#e74c3c', label='Supervised Methods')
    green_patch = mpatches.Patch(color='#2ecc71', label='VLM Methods')
    ax.legend(handles=[red_patch, green_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'inference_time_comparison.png'}")

def create_radar_chart(results):
    """Create radar chart comparing key metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC AUC']
    
    # Few-Shot VLM
    few_shot = results['vlm_1frame'][1]  # Few-Shot VLM
    few_shot_values = [
        few_shot['metrics']['accuracy'],
        few_shot['metrics']['precision'],
        few_shot['metrics']['recall'],
        few_shot['metrics']['f1_score'],
        few_shot['metrics']['specificity'],
        few_shot['metrics']['roc_auc']
    ]
    
    # Zero-Shot VLM
    zero_shot = results['vlm_1frame'][0]  # Zero-Shot VLM
    zero_shot_values = [
        zero_shot['metrics']['accuracy'],
        zero_shot['metrics']['precision'],
        zero_shot['metrics']['recall'],
        zero_shot['metrics']['f1_score'],
        zero_shot['metrics']['specificity'],
        zero_shot['metrics']['roc_auc']
    ]
    
    # 2D CNN
    cnn_2d = results['supervised'][3]  # 2D CNN
    cnn_2d_values = [
        cnn_2d['metrics']['accuracy'],
        cnn_2d['metrics']['precision'],
        cnn_2d['metrics']['recall'],
        cnn_2d['metrics']['f1_score'],
        cnn_2d['metrics']['specificity'],
        cnn_2d['metrics']['roc_auc']
    ]
    
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Complete values
    few_shot_values += few_shot_values[:1]
    zero_shot_values += zero_shot_values[:1]
    cnn_2d_values += cnn_2d_values[:1]
    
    # Plot
    ax.plot(angles, few_shot_values, 'o-', linewidth=2, label='Few-Shot VLM', color='#2ecc71')
    ax.fill(angles, few_shot_values, alpha=0.25, color='#2ecc71')
    
    ax.plot(angles, zero_shot_values, 'o-', linewidth=2, label='Zero-Shot VLM', color='#3498db')
    ax.fill(angles, zero_shot_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, cnn_2d_values, 'o-', linewidth=2, label='2D CNN', color='#e74c3c')
    ax.fill(angles, cnn_2d_values, alpha=0.25, color='#e74c3c')
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    ax.set_title('Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'radar_chart.png'}")

def main():
    """Generate all visualizations."""
    print("Generating report visualizations...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    results = load_results()
    
    create_comparison_bar_chart(results)
    create_confusion_matrices(results)
    create_inference_time_comparison(results)
    create_radar_chart(results)
    
    print("\n✓ All visualizations generated successfully!")
    print(f"Figures saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


