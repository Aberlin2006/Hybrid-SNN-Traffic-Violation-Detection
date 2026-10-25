import torch, os, sys, cv2, numpy as np
sys.path.insert(0, '.')
from training.dataset_pipeline import ProceduralTrafficGenerator

os.makedirs('tools', exist_ok=True)

print('=== DIAGNOSTIC: What SNN was trained on for CLASS 0 (Car) ===')
for i in range(5):
    img = ProceduralTrafficGenerator.generate_sample(0)
    cv2.imwrite(f'tools/train_car_sample_{i}.jpg', img)
    print(f'  Car sample {i}: shape={img.shape}, mean={img.mean():.1f}')

print('\n=== CHECKPOINT CLASS BIAS ANALYSIS ===')
ckpt = torch.load('vectorblox_deployment/artifacts/best_snn_checkpoint.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
w1 = sd['hidden_layer.weight']  # [64, 6912]
w2 = sd['output_layer.weight']  # [10, 64]
print(f'W2 (output layer) row norms per class:')
for i in range(10):
    row_norm = float(w2[i].norm())
    row_sum  = float(w2[i].sum())
    print(f'  Class {i}: norm={row_norm:.4f}  sum={row_sum:.4f}')

meta = ckpt.get('metrics', ckpt.get('epoch', 'N/A'))
print(f'\nCheckpoint metadata: {meta}')
acc = ckpt.get('accuracy', ckpt.get('val_accuracy', 'not stored'))
print(f'Stored accuracy: {acc}')
