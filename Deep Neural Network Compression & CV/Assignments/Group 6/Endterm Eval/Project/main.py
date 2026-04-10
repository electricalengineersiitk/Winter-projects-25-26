import config
from data.dataset import build_mnist_dataloaders
from models.mlp import MNIST_MLP
from utils.training import train_model, evaluate


from compression.pruning import (
    apply_magnitude_pruning, 
    make_pruning_permanent, 
    get_global_sparsity,
    print_sparsity_stats,
    print_memory_footprint 
)
from compression.quantization import apply_weight_sharing, print_quantization_stats
from compression.huffman import apply_huffman_encoding

def main():
    print("=====================================")
    print("    Deep Compression Pipeline v1.0   ")
    print("=====================================")
    print(f"[*] Computation Device: {config.DEVICE}")

    
    # PHASE 1: BASELINE
    
    print("\n[*] Loading MNIST Dataset...")
    train_loader, test_loader = build_mnist_dataloaders(config.BATCH_SIZE)

    print("[*] Initializing Baseline MNIST_MLP Model...")
    model = MNIST_MLP()

    print(f"\n[*] Starting Baseline Training ({config.EPOCHS} Epochs)...")
    train_model(model, train_loader, config.EPOCHS, config.LEARNING_RATE, config.DEVICE)

    print("\n[*] Evaluating Baseline Model...")
    baseline_acc = evaluate(model, test_loader, config.DEVICE)
    print(f"[*] BASELINE ACCURACY: {baseline_acc:.2f}%")

    
    
    
    print("\n=====================================")
    print("        PHASE 2: PRUNING            ")
    print("=====================================")
    
    
    print_sparsity_stats(model, phase_name="BEFORE PRUNING")
    
    
    model = apply_magnitude_pruning(model, config.PRUNING_THRESHOLD)
    
    
    print_sparsity_stats(model, phase_name="AFTER PRUNING")
    
   
    sparsity = get_global_sparsity(model)
    print(f"[*] Global Sparsity achieved: {sparsity * 100:.2f}%")
    
   
    print("\n[*] Accuracy immediately after pruning (before healing):")
    dropped_acc = evaluate(model, test_loader, config.DEVICE)
    print(f"[*] DROPPED ACCURACY: {dropped_acc:.2f}%")
    
    
    fine_tune_epochs = max(1, config.EPOCHS // 2) 
    fine_tune_lr = config.LEARNING_RATE / 10
    
    print(f"\n[*] Fine-tuning pruned model to recover accuracy ({fine_tune_epochs} Epochs).")
    train_model(model, train_loader, fine_tune_epochs, fine_tune_lr, config.DEVICE)
    
   
    print("\n[*] Evaluating Pruned & Fine-Tuned Model...")
    pruned_acc = evaluate(model, test_loader, config.DEVICE)
    
    
    model = make_pruning_permanent(model)

   
    print("\n=====================================")
    print("        PIPELINE CHECKPOINT         ")
    print("=====================================")
    print(f"[*] BASELINE ACCURACY: {baseline_acc:.2f}%")
    print(f"[*] PRUNED ACCURACY:   {pruned_acc:.2f}%")
    print(f"[*] NETWORK SPARSITY:  {sparsity * 100:.2f}%")
    print("=====================================")
    
    
    
    print("\n=====================================")
    print("        PHASE 3: QUANTIZATION       ")
    print("=====================================")
    
   
    print("\n[*] Before Quantization:")
    print_quantization_stats(model)
    
    
    bits = getattr(config, 'QUANTIZATION_BITS', 4) 
    model = apply_weight_sharing(model, bits=bits)
    
   
    print("\n[*] After Quantization:")
    print_quantization_stats(model)
    
    
    print("\n[*] Evaluating Final Compressed Model...")
    final_acc = evaluate(model, test_loader, config.DEVICE)
    
    # --- FINAL DEEP COMPRESSION METRICS ---
    print("\n==================================================")
    print("          DEEP COMPRESSION FINAL REPORT         ")
    print("==================================================")
    print(f"[*] Original Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"[*] Pruned (Fine-Tuned) Acc:    {pruned_acc:.2f}%")
    print(f"[*] Final Quantized Accuracy:   {final_acc:.2f}%")
    print(f"[*] Total Network Sparsity:     {sparsity * 100:.2f}%")
    print(f"[*] Quantization Level:         {bits}-bit ({2**bits} weights/layer)")
    print("==================================================")

    
    print("\n[*] Evaluating Final Compressed Model...")
    final_acc = evaluate(model, test_loader, config.DEVICE)
    
    # --- FINAL DEEP COMPRESSION METRICS ---
    print("\n==================================================")
    print("          DEEP COMPRESSION FINAL REPORT         ")
    print("==================================================")
    print(f"[*] Original Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"[*] Pruned (Fine-Tuned) Acc:    {pruned_acc:.2f}%")
    print(f"[*] Final Quantized Accuracy:   {final_acc:.2f}%")
    print(f"[*] Total Network Sparsity:     {sparsity * 100:.2f}%\n")
    
    # Calculate Memory Storage
    print("[*] --- STORAGE SAVINGS ---")
    
    # Original model uses 100% of weights at 32-bits (Float32)
    # We pass a fresh MNIST_MLP to get the true baseline size without pruned zeros
    baseline_model = MNIST_MLP() 
    original_size = print_memory_footprint(baseline_model, bits=32, phase_name="Baseline")
    
    # Compressed model uses only active weights at 4-bits (from config)
    compressed_size = print_memory_footprint(model, bits=bits, phase_name="Compressed")
    
    # Calculate Compression Ratio (CR)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    print(f"   [-] Compression Ratio: {compression_ratio:.1f}x smaller!")
    print("==================================================")
   
    
    print("\n=====================================")
    print("        PHASE 4: HUFFMAN ENCODING   ")
    print("=====================================")
    huffman_size_mb = apply_huffman_encoding(model)

   
    print("[*] Evaluating Final Compressed Model...")
    final_acc = evaluate(model, test_loader, config.DEVICE)
    
    # --- FINAL DEEP COMPRESSION METRICS ---
    print("\n==================================================")
    print("          DEEP COMPRESSION FINAL REPORT         ")
    print("==================================================")
    print(f"[*] Original Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"[*] Pruned (Fine-Tuned) Acc:    {pruned_acc:.2f}%")
    print(f"[*] Final Quantized Accuracy:   {final_acc:.2f}%")
    print(f"[*] Total Network Sparsity:     {sparsity * 100:.2f}%\n")
    
    # Calculate Memory Storage
    print("[*] --- STORAGE SAVINGS ---")
    
    baseline_model = MNIST_MLP() 
    original_size = print_memory_footprint(baseline_model, bits=32, phase_name="Baseline")
    compressed_size = print_memory_footprint(model, bits=bits, phase_name="Quantized")
    
    # NEW: Show the Huffman size
    print(f"   [-] Huffman Encoded Size: {huffman_size_mb:.4f} MB (Variable-bit)")
    
    # Calculate Final Compression Ratio (CR)
    final_compression_ratio = original_size / huffman_size_mb if huffman_size_mb > 0 else 0
    print(f"\n   [!!!] FINAL COMPRESSION RATIO: {final_compression_ratio:.1f}x smaller [!!!]")
    print("==================================================")

if __name__ == "__main__":
    main()