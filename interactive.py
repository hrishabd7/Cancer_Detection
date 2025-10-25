import torch
import matplotlib.pyplot as plt
import random
from PIL import Image

def interactive_predict(model, dataset, device, class_names, eval_tf):
    """Pop up random images and show model prediction."""
    model.eval()
    while True:
        idx = random.randint(0, len(dataset) - 1)
        img_path, true_label = dataset.samples[idx]
        img = Image.open(img_path)
        img_tensor = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = logits.argmax(1).item()
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_idx]} ({probs[pred_idx]:.2f})")
        plt.axis('off')
        plt.show()
        cont = input("Show another image? (y/n): ")
        if cont.lower() != 'y':
            break