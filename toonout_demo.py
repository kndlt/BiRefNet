#!/usr/bin/env python3
"""
ToonOut Demo Script
Remove background from images using fine-tuned BiRefNet model.

Usage:
    python toonout_demo.py --weights path/to/weights.pth --input path/to/image.jpg [--output result.png]
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
import sys
import os
from huggingface_hub import hf_hub_download

# Simple fix for BiRefNet compatibility
import transformers.configuration_utils
original_getattribute = transformers.configuration_utils.PretrainedConfig.__getattribute__

def patched_getattribute(self, key):
    if key == 'is_encoder_decoder':
        return False
    return original_getattribute(self, key)

transformers.configuration_utils.PretrainedConfig.__getattribute__ = patched_getattribute

from transformers import AutoModelForImageSegmentation


def load_birefnet_with_custom_weights(checkpoint_path: str = None):
    """Load BiRefNet model with custom fine-tuned weights"""
    
    print(f"Loading base BiRefNet model from HuggingFace...")
    # Load the base model from HuggingFace
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        trust_remote_code=True
    )
    
    # Download weights from HuggingFace if no local path provided
    if checkpoint_path is None:
        print("Downloading fine-tuned weights from HuggingFace (joelseytre/toonout)...")
        checkpoint_path = hf_hub_download(
            repo_id="joelseytre/toonout",
            filename="birefnet_finetuned_toonout.pth"
        )
    
    print(f"Loading custom weights from {checkpoint_path}...")
    # Load and apply custom weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Clean up weight keys if needed (remove module prefixes)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module._orig_mod."):
            clean_state_dict[k[len("module._orig_mod."):]] = v
        elif k.startswith("module."):
            clean_state_dict[k[len("module."):]] = v
        else:
            clean_state_dict[k] = v
    
    model.load_state_dict(clean_state_dict)
    print("Model loaded successfully!")
    return model


def remove_background(image_path: str, model, device='cpu'):
    """Remove background from image using BiRefNet"""
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Processing image: {image_path}")
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate mask
    print("Generating mask...")
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Convert mask to PIL and resize to original size
    mask = transforms.ToPILImage()(preds[0].squeeze())
    mask = mask.resize(image.size)
    
    # Apply mask to create transparent background
    result = image.copy()
    result.putalpha(mask)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Remove background from images using ToonOut (fine-tuned BiRefNet)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-download weights from HuggingFace
  python toonout_demo.py --input test.jpg
  
  # Use local weights file
  python toonout_demo.py --weights model.pth --input test.jpg
  python toonout_demo.py -w model.pth -i test.jpg -o result.png
        """
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to the fine-tuned BiRefNet weights (.pth file). If not provided, will auto-download from HuggingFace (joelseytre/toonout)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='sample.png',
        help='Path to the input image (default: sample.png)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save the output image (default: input_name_nobg.png)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.weights and not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_birefnet_with_custom_weights(args.weights)
    
    try:
        model.to(device)
        model.eval()
        
        # Process image
        result = remove_background(args.input, model, device)
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            print(f"\nWarning: CUDA error encountered: {str(e)[:100]}...")
            print("Falling back to CPU...\n")
            device = "cpu"
            model.to(device)
            model.eval()
            result = remove_background(args.input, model, device)
        else:
            raise
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_nobg.png"
    else:
        output_path = args.output
    
    # Save result
    result.save(output_path)
    print(f"✓ Background removed successfully!")
    print(f"✓ Result saved to: {output_path}")


if __name__ == "__main__":
    main()
