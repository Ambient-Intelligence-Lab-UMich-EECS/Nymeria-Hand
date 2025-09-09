import os
import json
import argparse
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from skimage import measure
import warnings
import subprocess
import pandas as pd
warnings.filterwarnings('ignore')


class LabelmeSegmentationDataset(Dataset):
    """Dataset for loading images and labelme polygon masks for SegFormer training."""
    
    def __init__(self, data_root, transform=None, image_processor=None):
        """
        Initialize dataset with labelme annotations.
        
        Args:
            data_root: Root directory containing subdirectories with PNG images and JSON labelme files
            transform: Albumentations transform pipeline
            image_processor: SegFormer image processor for preprocessing
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.image_processor = image_processor
        
        # Class mapping: 0=background, 1=arm-left, 2=arm-right, 3=person (negative class)
        self.class_names = ['background', 'arm-left', 'arm-right', 'person']
        self.num_classes = len(self.class_names)
        
        # Find all PNG files and corresponding JSON files, filter for arm annotations
        self.samples = []
        skipped_count = 0
        synthesis_count = 0
        synthesis_kept = 0
        background_count = 0
        
        # First, collect all annotated samples
        for png_file in self.data_root.rglob('*.png'):
            json_file = png_file.with_suffix('.json')
            if json_file.exists():
                # Check if this is from a synthesis dataset
                is_synthesis = 'synthesis' in str(png_file).lower()
                
                # If synthesis, randomly downsample to 10%
                if is_synthesis:
                    synthesis_count += 1
                    if random.random() > 0.1:  # Skip 90% of synthesis data
                        continue
                    synthesis_kept += 1
                
                # Check if JSON contains arm annotations
                if self._has_arm_annotations(json_file):
                    self.samples.append((str(png_file), str(json_file), False))  # False = not background
                else:
                    skipped_count += 1
        
        # Now add pure background samples from background.txt files
        for bg_txt_file in self.data_root.rglob('background.txt'):
            parent_dir = bg_txt_file.parent
            try:
                with open(bg_txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            png_path = parent_dir / line
                            if png_path.exists():
                                self.samples.append((str(png_path), None, True))  # True = pure background
                                background_count += 1
            except Exception as e:
                print(f"Warning: Could not read background.txt in {parent_dir}: {e}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image-annotation pairs found in {data_root}")
        
        print(f"Dataset: {len(self.samples)} total samples")
        print(f"  - Annotated frames: {len(self.samples) - background_count}")
        print(f"  - Pure background frames: {background_count}")
        if synthesis_count > 0:
            print(f"  - Synthesis data: {synthesis_kept}/{synthesis_count} samples kept (10% downsampling)")
    
    def _has_arm_annotations(self, json_path):
        """
        Check if JSON file contains arm annotations.
        
        Args:
            json_path: Path to labelme JSON file
            
        Returns:
            True if JSON contains arm-left or arm-right annotations
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if any shape has arm or person labels
            for shape in data.get('shapes', []):
                label = shape['label']
                if label in ['arm-left', 'left-arm', 'arm-right', 'right-arm', 'person']:
                    return True
            return False
            
        except Exception:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, json_path, is_background = self.samples[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Create mask: pure background or parse labelme JSON
        if is_background:
            # Pure background frame - all pixels are class 0
            mask = np.zeros((original_height, original_width), dtype=np.uint8)
        else:
            # Load and parse labelme JSON
            mask = self.parse_labelme_json(json_path, original_height, original_width)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Process with SegFormer processor if provided
        if self.image_processor:
            # Convert back to PIL for processor
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Process image
            inputs = self.image_processor(image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            
            # Resize mask to match processed image size
            processed_size = pixel_values.shape[-2:]  # (H, W)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            mask = cv2.resize(mask.astype(np.float32), 
                            (processed_size[1], processed_size[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int64)
            mask = torch.from_numpy(mask)
            
            return {
                'pixel_values': pixel_values,
                'labels': mask,
                'original_size': (original_height, original_width)
            }
        
        return {
            'image': image,
            'mask': mask,
            'original_size': (original_height, original_width)
        }
    
    def parse_labelme_json(self, json_path, height, width):
        """
        Parse labelme JSON and create segmentation mask.
        
        Args:
            json_path: Path to labelme JSON file
            height, width: Image dimensions
            
        Returns:
            Segmentation mask (H, W) with class indices
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process each shape
        for shape in data.get('shapes', []):
            label = shape['label']
            points = shape['points']
            
            # Map label to class index (handle both naming conventions)
            if label in ['arm-left', 'left-arm']:
                class_idx = 1
            elif label in ['arm-right', 'right-arm']:
                class_idx = 2
            elif label == 'person':
                class_idx = 3
            else:
                continue  # Skip unknown labels
            
            # Create polygon mask using cv2.fillPoly (more reliable than PIL)
            if len(points) >= 3:  # Need at least 3 points for polygon
                # Convert points to numpy array format for cv2
                polygon_points = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
                
                # Create temporary mask for this polygon
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Fill polygon
                cv2.fillPoly(temp_mask, [polygon_points], class_idx)
                
                # Check polygon size - if less than 
                #  pixels, skip it (treat as background)
                polygon_size = np.sum(temp_mask == class_idx)
                if polygon_size < 500:
                    continue  # Skip small polygons
                
                # Add to main mask (later labels will overwrite earlier ones at overlapping pixels)
                mask[temp_mask == class_idx] = class_idx
        
        return mask


class SegFormerTrainer:
    """SegFormer training and inference pipeline for arm segmentation."""
    
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=4, device=None):
        """
        Initialize SegFormer trainer.
        
        Args:
            model_name: Pretrained SegFormer model name
            num_classes: Number of classes (background + arm classes + person)
            device: Device to use for training/inference
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = ['background', 'arm-left', 'arm-right', 'person']
        
        # Load model and processor
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        
        # Define class colors for visualization
        self.colors = {
            0: (0, 0, 0),       # background - black
            1: (255, 0, 0),     # arm-left - red
            2: (0, 255, 0),     # arm-right - green
            3: (0, 0, 255),     # person - blue (not shown in inference)
        }
    
    def get_transforms(self, is_training=True):
        """Get data augmentation transforms."""
        if is_training:
            return A.Compose([
                A.Resize(512, 512),
                A.ShiftScaleRotate(
                    shift_limit=0.03, scale_limit=0.2, rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7
                ),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.7, contrast_limit=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=(2, 5), p=1.0),
                    A.GaussNoise(var_limit=(5.0, 10.0), p=1.0),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def train(self, data_root, output_dir, epochs=50, batch_size=4, learning_rate=5e-5, 
              val_split=0.2, save_every=10):
        """
        Train SegFormer model on the dataset.
        
        Args:
            data_root: Root directory with training data
            output_dir: Directory to save model checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            save_every: Save model every N epochs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        full_dataset = LabelmeSegmentationDataset(
            data_root, 
            transform=self.get_transforms(is_training=True),
            image_processor=self.image_processor
        )
        
        # Split dataset
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Training: {train_size} samples, Validation: {val_size} samples")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Get model outputs with CE loss
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc="Validation")
                for batch in pbar:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    val_steps += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_val_loss = val_loss / val_steps
            scheduler.step()
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(output_dir / "best_model")
                print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(output_dir / f"checkpoint_epoch_{epoch+1}")
                print(f"Checkpoint saved at epoch {epoch+1}")
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, save_path):
        """Save model and processor."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
    
    def load_model(self, model_path):
        """Load trained model."""
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.image_processor = SegformerImageProcessor.from_pretrained(model_path)
    
    def predict_image(self, image, return_probs=False):
        """
        Predict segmentation mask for a single image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probs: Whether to return probability maps
            
        Returns:
            Segmentation mask and optionally probability maps
        """
        self.model.eval()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        original_size = image.size  # (W, H)
        
        # Process image
        inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Resize logits to original image size
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=original_size[::-1],  # (H, W)
                mode="bilinear",
                align_corners=False
            )
            
            # Get predictions
            probs = torch.softmax(upsampled_logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Convert person class (3) to background (0) for inference
            pred_mask[pred_mask == 3] = 0
            
            # Filter out small regions (< 500 pixels)
            filtered_mask = np.zeros_like(pred_mask)
            for class_idx in [1, 2]:  # Only process arm classes
                if class_idx not in np.unique(pred_mask):
                    continue
                    
                # Create binary mask for this class
                class_mask = (pred_mask == class_idx).astype(np.uint8)
                
                # Find connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
                
                # Keep only components larger than 500 pixels
                for label_idx in range(1, num_labels):  # Skip background (0)
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if area >= 500:
                        filtered_mask[labels == label_idx] = class_idx
            
            # Return filtered mask
            if return_probs:
                probs = probs.squeeze().cpu().numpy()
                return filtered_mask, probs
            
            return filtered_mask
    
    def predict_batch(self, images, original_sizes, return_probs=False):
        """
        Predict segmentation masks for a batch of images.
        
        Args:
            images: List of input images (numpy arrays)
            original_sizes: List of original image sizes (W, H)
            return_probs: Whether to return probability maps
            
        Returns:
            List of segmentation masks and optionally probability maps
        """
        self.model.eval()
        
        # Convert all images to PIL and process
        pil_images = []
        for image in images:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            pil_images.append(image)
        
        # Process all images in batch
        inputs = self.image_processor(pil_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # Shape: (batch_size, num_classes, H, W)
            
            # Process each image in the batch
            pred_masks = []
            all_probs = [] if return_probs else None
            
            for i in range(logits.shape[0]):
                # Get logits for this image
                single_logits = logits[i:i+1]  # Keep batch dimension
                original_size = original_sizes[i]
                
                # Resize logits to original image size
                upsampled_logits = nn.functional.interpolate(
                    single_logits,
                    size=original_size[::-1],  # (H, W)
                    mode="bilinear",
                    align_corners=False
                )
                
                # Get predictions
                probs = torch.softmax(upsampled_logits, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                # Convert person class (3) to background (0) for inference
                pred_mask[pred_mask == 3] = 0
                
                # Filter out small regions (< 500 pixels)
                filtered_mask = np.zeros_like(pred_mask)
                for class_idx in [1, 2]:  # Only process arm classes
                    if class_idx not in np.unique(pred_mask):
                        continue
                        
                    # Create binary mask for this class
                    class_mask = (pred_mask == class_idx).astype(np.uint8)
                    
                    # Find connected components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
                    
                    # Keep only components larger than 500 pixels
                    for label_idx in range(1, num_labels):  # Skip background (0)
                        area = stats[label_idx, cv2.CC_STAT_AREA]
                        if area >= 500:
                            filtered_mask[labels == label_idx] = class_idx
                
                pred_masks.append(filtered_mask)
                
                if return_probs:
                    all_probs.append(probs.squeeze().cpu().numpy())
            
            if return_probs:
                return pred_masks, all_probs
            
            return pred_masks
    
    def create_overlay(self, image, mask, alpha=0.2):
        """
        Create overlay visualization of segmentation mask.
        
        Args:
            image: Original image (numpy array in RGB format)
            mask: Segmentation mask (uint8)
            alpha: Transparency factor
            
        Returns:
            Overlay image in RGB format
        """
        if isinstance(image, np.ndarray):
            overlay = image.copy().astype(np.uint8)
        else:
            overlay = np.array(image, dtype=np.uint8)
        
        # Create colored mask (only show arms in inference, not background)
        # Colors are in RGB format since we're working with RGB images
        rgb_colors = {
            0: (0, 0, 0),       # background - black
            1: (255, 0, 0),     # arm-left - red
            2: (0, 255, 0),     # arm-right - green
        }
        
        colored_mask = np.zeros_like(overlay, dtype=np.uint8)
        for class_idx, color in rgb_colors.items():
            if class_idx in [1, 2]:  # Only show arm-left and arm-right
                colored_mask[mask == class_idx] = color
        
        # Add mask on top without darkening original
        result = overlay.copy()
        mask_pixels = colored_mask.any(axis=2)  # Find pixels with any color
        if mask_pixels.any():  # Check if there are any mask pixels
            result[mask_pixels] = cv2.addWeighted(
                overlay[mask_pixels], 1 - alpha, 
                colored_mask[mask_pixels], alpha, 0
            )
        
        return result

    def mask_to_labelme_json(self, mask, image_width, image_height, frame_filename="frame.png"):
        """
        Convert segmentation mask to labelme JSON format.
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            image_width: Original image width
            image_height: Original image height
            frame_filename: Name of the corresponding image file
            
        Returns:
            Dictionary in labelme JSON format
        """
        # Create labelme JSON structure
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": frame_filename,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }
        
        # Process each class (skip background class 0)
        for class_idx in [1, 2]:  # arm-left, arm-right
            if class_idx not in np.unique(mask):
                continue
                
            class_name = self.class_names[class_idx]
            
            # Create binary mask for this class
            class_mask = (mask == class_idx).astype(np.uint8)
            
            # Find contours using skimage
            contours = measure.find_contours(class_mask, 0.5)
            
            for contour in contours:
                # Skip very small contours (noise)
                if len(contour) < 10:
                    continue
                
                # Convert contour to labelme format (x, y coordinates)
                # Note: measure.find_contours returns (row, col) so we need to swap
                points = [[float(point[1]), float(point[0])] for point in contour]
                
                # Close the polygon by adding the first point at the end if needed
                if len(points) > 0 and points[0] != points[-1]:
                    points.append(points[0])
                
                shape = {
                    "label": class_name,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                
                labelme_data["shapes"].append(shape)
        
        return labelme_data

    def add_frame_number_overlay(self, image, frame_number, font_scale=0.7, thickness=2):
        """
        Add frame number overlay to the corner of the image.
        
        Args:
            image: Input image (BGR format for OpenCV)
            frame_number: Frame number to display
            font_scale: Font scale factor
            thickness: Text thickness
            
        Returns:
            Image with frame number overlay
        """
        # Create a copy to avoid modifying the original
        img_with_overlay = image.copy()
        
        # Frame number text
        frame_text = f"Frame: {frame_number}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)     # Black background
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
        
        # Position in top-left corner with some padding
        padding = 10
        x = padding
        y = padding + text_height
        
        # Draw background rectangle
        cv2.rectangle(img_with_overlay, 
                     (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(img_with_overlay, frame_text, (x, y), font, font_scale, color, thickness)
        
        return img_with_overlay

        
    def process_video(self, input_video, output_video, fps=None, show_overlay=True, batch_size=16, use_batch_inference=True, output_size=(320, 320), use_ffmpeg=True, save_frames_dir=None):
        """
        Process video and generate segmentation output.
        
        Args:
            input_video: Path to input video
            output_video: Path to output video
            fps: Output FPS (None to use original)
            show_overlay: Whether to show overlay or just segmentation
            batch_size: Number of frames to process in each batch (only used if use_batch_inference=True)
            use_batch_inference: Whether to use batch inference or process frames individually
            output_size: Output video size as (width, height) tuple
            use_ffmpeg: Whether to use ffmpeg for faster video writing
            save_frames_dir: Directory to save individual frames as PNG and labelme JSON files (optional)
        """
        # Open input video
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps is None:
            fps = original_fps
        
        # Calculate frame sampling rate for downsampling during reading
        frame_skip = 1
        if fps < original_fps:
            frame_skip = int(original_fps / fps)
            actual_output_fps = original_fps / frame_skip
            frames_to_process = total_frames // frame_skip
        else:
            actual_output_fps = fps
            frames_to_process = total_frames
        
        print(f"Processing video: {input_video}")
        print(f"Input resolution: {frame_width}x{frame_height}")
        print(f"Processing resolution: 512x512 (SegFormer input)")
        print(f"Output resolution: {output_size[0]}x{output_size[1]}")
        print(f"Input FPS: {original_fps}, Output FPS: {actual_output_fps}")
        print(f"Total input frames: {total_frames}")
        if frame_skip > 1:
            print(f"Frame sampling: every {frame_skip} frames (downsampling from {original_fps} to {actual_output_fps} fps)")
            print(f"Frames to process: {frames_to_process}")
        print(f"Video writer: {'ffmpeg' if use_ffmpeg else 'opencv'}")
        print(f"Batch inference: {'enabled' if use_batch_inference else 'disabled'}")
        if use_batch_inference:
            print(f"Batch size: {batch_size}")
        
        # Read frames with downsampling if needed
        print("Loading frames into memory...")
        frames = []
        frame_count = 0
        with tqdm(total=frames_to_process, desc="Loading frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only keep frames according to sampling rate
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB and resize to 512x512 for SegFormer processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_512 = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    frames.append(frame_512)
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        print(f"Loaded {len(frames)} frames")
        
        # Process frames and store masks in memory
        print("Running inference...")
        masks = []
        
        if use_batch_inference and len(frames) > 0:
            # Batch inference - use 512x512 for processing
            processing_sizes = [(512, 512)] * len(frames)
            
            with tqdm(total=len(frames), desc="Batch inference") as pbar:
                for i in range(0, len(frames), batch_size):
                    batch_frames = frames[i:i + batch_size]
                    batch_sizes = processing_sizes[i:i + batch_size]
                    
                    # Process batch at 512x512
                    batch_masks = self.predict_batch(batch_frames, batch_sizes)
                    masks.extend(batch_masks)
                    
                    pbar.update(len(batch_frames))
        else:
            # Individual frame inference - frames are already 512x512
            with tqdm(total=len(frames), desc="Individual inference") as pbar:
                for frame in frames:
                    mask = self.predict_image(frame)
                    masks.append(mask)
                    pbar.update(1)
        
        print(f"Inference complete. Generated {len(masks)} masks")
        
        # Save individual frames and labelme JSON files if directory is provided
        if save_frames_dir:
            save_frames_path = Path(save_frames_dir)
            save_frames_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving frames and JSON files to {save_frames_dir}...")
            with tqdm(total=len(frames), desc="Saving frames") as pbar:
                for i, (frame_rgb, mask) in enumerate(zip(frames, masks)):
                    # Format frame number with leading zeros (e.g., 0000, 0001, etc.)
                    frame_num = f"{i:05d}"
                    
                    # Save PNG frame
                    png_filename = f"{frame_num}.png"
                    png_path = save_frames_path / png_filename
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(png_path), frame_bgr)
                    
                    # Create and save labelme JSON
                    json_filename = f"{frame_num}.json"
                    json_path = save_frames_path / json_filename
                    
                    # Use original frame size (512x512 since we resized for processing)
                    labelme_data = self.mask_to_labelme_json(
                        mask, 
                        image_width=512, 
                        image_height=512, 
                        frame_filename=png_filename
                    )
                    
                    with open(json_path, 'w') as f:
                        json.dump(labelme_data, f, indent=2)
                    
                    pbar.update(1)
            
            print(f"Saved {len(frames)} frames and JSON files to {save_frames_dir}")
        
        if use_ffmpeg:
            # Use ffmpeg for faster video writing
            self._write_video_ffmpeg(frames, masks, output_video, actual_output_fps, output_size, show_overlay)
        else:
            # Use OpenCV video writer (slower but more compatible)
            self._write_video_opencv(frames, masks, output_video, actual_output_fps, output_size, show_overlay)
        
        print(f"Video processing complete! Output saved to {output_video}")
    
    def _write_video_opencv(self, frames, masks, output_video, fps, output_size, show_overlay):
        """Write video using OpenCV VideoWriter (slower but more compatible)."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, output_size)
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_video}")
            return
        
        print("Writing video with OpenCV...")
        with tqdm(total=len(frames), desc="Writing video") as pbar:
            for frame_idx, (frame_rgb, mask) in enumerate(zip(frames, masks)):
                # Resize frame and mask from 512x512 to output size
                frame_resized = cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
                
                if show_overlay:
                    result_frame = self.create_overlay(frame_resized, mask_resized)
                else:
                    result_frame = np.zeros_like(frame_resized, dtype=np.uint8)
                    for class_idx, color in self.colors.items():
                        result_frame[mask_resized == class_idx] = color
                
                # Convert back to BGR for video writing
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                # Add frame number overlay
                result_frame_bgr = self.add_frame_number_overlay(result_frame_bgr, frame_idx)
                
                out.write(result_frame_bgr)
                pbar.update(1)
        
        out.release()
    
    def _write_video_ffmpeg(self, frames, masks, output_video, fps, output_size, show_overlay):
        """Write video using ffmpeg pipe (faster)."""
        # FFmpeg command for faster encoding
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{output_size[0]}x{output_size[1]}', '-pix_fmt', 'rgb24',
            '-r', str(fps), '-i', '-', '-an',
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18', '-preset', 'fast',
            str(output_video)
        ]
        
        try:
            # Start ffmpeg process
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("Writing video with ffmpeg...")
            with tqdm(total=len(frames), desc="Writing video") as pbar:
                for frame_idx, (frame_rgb, mask) in enumerate(zip(frames, masks)):
                    # Resize frame and mask from 512x512 to output size
                    frame_resized = cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_LINEAR)
                    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
                    
                    if show_overlay:
                        result_frame = self.create_overlay(frame_resized, mask_resized)
                    else:
                        result_frame = np.zeros_like(frame_resized, dtype=np.uint8)
                        for class_idx, color in self.colors.items():
                            result_frame[mask_resized == class_idx] = color
                    
                    # Convert to BGR for frame number overlay, then back to RGB
                    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    result_frame_bgr = self.add_frame_number_overlay(result_frame_bgr, frame_idx)
                    result_frame = cv2.cvtColor(result_frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Write raw RGB frame to ffmpeg stdin
                    proc.stdin.write(result_frame.tobytes())
                    pbar.update(1)
            
            # Close stdin and wait for process to finish
            proc.stdin.close()
            proc.wait()
            
            if proc.returncode != 0:
                stderr_output = proc.stderr.read().decode()
                print(f"FFmpeg error: {stderr_output}")
                print("Falling back to OpenCV video writer...")
                self._write_video_opencv(frames, masks, output_video, fps, output_size, show_overlay)
                
        except FileNotFoundError:
            print("FFmpeg not found. Falling back to OpenCV video writer...")
            self._write_video_opencv(frames, masks, output_video, fps, output_size, show_overlay)
        except Exception as e:
            print(f"FFmpeg error: {e}")
            print("Falling back to OpenCV video writer...")
            self._write_video_opencv(frames, masks, output_video, fps, output_size, show_overlay)
    
    def process_image_file(self, input_image, output_image, save_overlay=True, save_mask=True):
        """
        Process a single image file and save segmentation results.
        
        Args:
            input_image: Path to input image file
            output_image: Path to save output image
            save_overlay: Whether to save overlay visualization
            save_mask: Whether to save segmentation mask
        """
        # Read image
        image = cv2.imread(str(input_image))
        if image is None:
            print(f"Error: Could not read image {input_image}")
            return
        
        print(f"Processing image: {input_image}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Predict segmentation
        mask = self.predict_image(image)
        
        output_path = Path(output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_overlay:
            # Create and save overlay
            overlay = self.create_overlay(image, mask)
            overlay_path = output_path.parent / f"{output_path.stem}_overlay{output_path.suffix}"
            cv2.imwrite(str(overlay_path), overlay)
            print(f"Overlay saved: {overlay_path}")
        
        if save_mask:
            # Create colored segmentation mask
            colored_mask = np.zeros_like(image)
            for class_idx, color in self.colors.items():
                colored_mask[mask == class_idx] = color
            
            mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
            cv2.imwrite(str(mask_path), colored_mask)
            print(f"Segmentation mask saved: {mask_path}")
            
            # Also save raw mask as grayscale
            raw_mask_path = output_path.parent / f"{output_path.stem}_raw_mask.png"
            cv2.imwrite(str(raw_mask_path), (mask * 85).astype(np.uint8))  # Scale for visibility
            print(f"Raw mask saved: {raw_mask_path}")
        
        # Print segmentation statistics
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        print("\nSegmentation Statistics:")
        for class_idx, count in zip(unique_classes, counts):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
            percentage = (count / total_pixels) * 100
            print(f"  {class_name}: {count} pixels ({percentage:.1f}%)")
    
    def process_image_folder(self, model_path, parent_folder, downsample_rate=1, batch_size=16):
        """
        Process a folder of PNG images with optional downsampling and save segmentation masks.
        
        Args:
            model_path: Path to trained model
            parent_folder: Parent folder containing camera-rgb-512x512 subfolder
            downsample_rate: Process every Nth image (1 = all images, 3 = every 3rd image)
            batch_size: Number of images to process in each batch
            
        Returns:
            Path to output folder containing segmentation masks
        """
        # Load model if provided path
        if model_path:
            self.load_model(model_path)
        
        parent_path = Path(parent_folder)
        if not parent_path.exists():
            raise ValueError(f"Parent folder does not exist: {parent_folder}")
        
        # Look for camera-rgb-512x512 subfolder
        input_path = parent_path / "camera-rgb-512x512"
        if not input_path.exists():
            raise ValueError(f"Input folder not found: {input_path}")
        
        # Create output folder at same level as camera-rgb-512x512
        output_folder = parent_path / "hand-segment"
        
        # Check if output folder exists and has files
        if output_folder.exists():
            existing_files = list(output_folder.glob("*.png"))
            if existing_files:
                print(f"Output folder {output_folder} already exists with {len(existing_files)} files. Skipping processing.")
                return str(output_folder)
        
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        # Look for timestamps.csv in the camera-rgb-512x512 folder
        timestamps_csv = input_path / "timestamps.csv"
        if not timestamps_csv.exists():
            raise ValueError(f"timestamps.csv not found in {input_path}")
        
        # Read timestamps CSV
        df = pd.read_csv(timestamps_csv)
        if 'png_filename' not in df.columns:
            raise ValueError(f"CSV file must have 'png_filename' column")
        
        # Get list of PNG files from CSV
        all_png_files = df['png_filename'].tolist()
        
        # Apply downsampling
        if downsample_rate > 1:
            # Select every Nth file based on index
            selected_indices = list(range(0, len(all_png_files), downsample_rate))
            selected_files = [all_png_files[i] for i in selected_indices]
            # Create downsampled dataframe
            df_downsampled = df.iloc[selected_indices].copy()
            print(f"Downsampling rate {downsample_rate}: selected {len(selected_files)} out of {len(all_png_files)} files")
        else:
            selected_files = all_png_files
            df_downsampled = df.copy()
            print(f"Processing all {len(selected_files)} files")
        
        # Save the (possibly downsampled) CSV to output folder
        output_csv = output_folder / "timestamps.csv"
        df_downsampled.to_csv(output_csv, index=False)
        print(f"Saved timestamps CSV to: {output_csv}")
        
        # Verify all selected files exist
        missing_files = []
        for filename in selected_files:
            filepath = input_path / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files not found in input folder")
            if len(missing_files) <= 10:
                for f in missing_files:
                    print(f"  - {f}")
            selected_files = [f for f in selected_files if f not in missing_files]
        
        if not selected_files:
            raise ValueError("No valid image files to process")
        
        print(f"\nProcessing {len(selected_files)} images with batch size {batch_size}")
        
        # Process images in batches
        total_processed = 0
        with tqdm(total=len(selected_files), desc="Processing images") as pbar:
            for batch_start in range(0, len(selected_files), batch_size):
                batch_end = min(batch_start + batch_size, len(selected_files))
                batch_files = selected_files[batch_start:batch_end]
                
                # Load batch of images
                batch_images = []
                batch_sizes = []
                valid_batch_files = []
                
                for filename in batch_files:
                    filepath = input_path / filename
                    
                    # Read image
                    image = cv2.imread(str(filepath))
                    if image is None:
                        print(f"Warning: Could not read {filepath}")
                        continue
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    batch_images.append(image_rgb)
                    batch_sizes.append((image.shape[1], image.shape[0]))  # (W, H)
                    valid_batch_files.append(filename)
                
                if not batch_images:
                    pbar.update(len(batch_files))
                    continue
                
                # Run batch inference
                try:
                    masks = self.predict_batch(batch_images, batch_sizes, return_probs=False)
                    
                    # Save each mask as RGB image
                    for filename, mask in zip(valid_batch_files, masks):
                        output_path = output_folder / filename
                        
                        # Create RGB image with white background for all non-arm regions
                        h, w = mask.shape
                        # Start with white background (255, 255, 255)
                        rgb_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
                        
                        # arm-left (class 1) = red
                        rgb_mask[mask == 1] = [255, 0, 0]
                        
                        # arm-right (class 2) = green  
                        rgb_mask[mask == 2] = [0, 255, 0]
                        
                        # Everything else (background, person, etc.) remains white
                        
                        # Save as BGR for OpenCV (so swap red and blue channels)
                        bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), bgr_mask)
                    
                    total_processed += len(valid_batch_files)
                    
                except Exception as e:
                    print(f"Error processing batch starting at {batch_start}: {e}")
                
                pbar.update(len(batch_files))
        
        print(f"\nProcessing complete!")
        print(f"Processed {total_processed} images")
        print(f"Output folder: {output_folder}")
        
        # Verify output files
        output_files = list(output_folder.glob("*.png"))
        print(f"Created {len(output_files)} mask files")
        
        # Create visualization video
        print(f"\nCreating visualization video...")
        self._create_mask_visualization_video(
            parent_path=parent_path,
            selected_files=selected_files,
            input_folder=input_path,
            mask_folder=output_folder,
            output_video=parent_path / "hand-segment-vis.mp4",
            downsample_rate=downsample_rate
        )
        
        return str(output_folder)
    
    def _create_mask_visualization_video(self, parent_path, selected_files, input_folder, mask_folder, output_video, downsample_rate=1, fps=30):
        """
        Create a visualization video combining original frames with mask overlays.
        
        Args:
            parent_path: Parent folder path
            selected_files: List of selected (possibly downsampled) file names
            input_folder: Path to original images folder
            mask_folder: Path to mask images folder
            output_video: Path for output video
            fps: Frames per second for output video
        """
        if not selected_files:
            print("No files to create video from")
            return
        
        # Read first frame to get dimensions
        first_file = selected_files[0]
        first_frame_path = input_folder / first_file
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            print(f"Warning: Could not read first frame for video creation")
            return
        
        height, width = first_frame.shape[:2]
        
        # Try to use ffmpeg for faster video writing
        use_ffmpeg = True
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except:
            use_ffmpeg = False
            print("FFmpeg not available, using OpenCV for video writing")
        
        if use_ffmpeg:
            # FFmpeg command for faster encoding
            cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'rgb24',
                '-r', str(fps), '-i', '-', '-an',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', '-preset', 'fast',
                str(output_video)
            ]
            
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                
                with tqdm(total=len(selected_files), desc="Writing visualization video") as pbar:
                    for idx, filename in enumerate(selected_files):
                        # Extract original frame number from filename (e.g., "00042.png" -> 42)
                        original_frame_number = int(filename.split('.')[0])
                        
                        # Read original frame
                        frame_path = input_folder / filename
                        frame = cv2.imread(str(frame_path))
                        
                        # Read mask
                        mask_path = mask_folder / filename
                        mask_img = cv2.imread(str(mask_path))
                        
                        if frame is None or mask_img is None:
                            print(f"Warning: Skipping {filename} - could not read frame or mask")
                            pbar.update(1)
                            continue
                        
                        # Convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                        
                        # Create overlay - blend original with mask
                        # Only overlay where mask is not white
                        white_mask = np.all(mask_rgb == [255, 255, 255], axis=2)
                        overlay = frame_rgb.copy()
                        
                        # Apply colored mask with transparency where it's not white
                        alpha = 0.3  # Transparency for mask overlay
                        overlay[~white_mask] = (
                            frame_rgb[~white_mask] * (1 - alpha) + 
                            mask_rgb[~white_mask] * alpha
                        ).astype(np.uint8)
                        
                        # Add frame number (use original frame number, not sequential index)
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        overlay_bgr = self.add_frame_number_overlay(overlay_bgr, original_frame_number)
                        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Write to ffmpeg
                        proc.stdin.write(overlay.tobytes())
                        pbar.update(1)
                
                proc.stdin.close()
                proc.wait()
                
                if proc.returncode == 0:
                    print(f"Visualization video saved to: {output_video}")
                else:
                    stderr_output = proc.stderr.read().decode()
                    print(f"FFmpeg error: {stderr_output}")
                    use_ffmpeg = False
                    
            except Exception as e:
                print(f"FFmpeg error: {e}")
                use_ffmpeg = False
        
        if not use_ffmpeg:
            # Fallback to OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Error: Could not create output video {output_video}")
                return
            
            with tqdm(total=len(selected_files), desc="Writing visualization video") as pbar:
                for idx, filename in enumerate(selected_files):
                    # Extract original frame number from filename (e.g., "00042.png" -> 42)
                    original_frame_number = int(filename.split('.')[0])
                    
                    # Read original frame
                    frame_path = input_folder / filename
                    frame = cv2.imread(str(frame_path))
                    
                    # Read mask
                    mask_path = mask_folder / filename
                    mask_img = cv2.imread(str(mask_path))
                    
                    if frame is None or mask_img is None:
                        print(f"Warning: Skipping {filename} - could not read frame or mask")
                        pbar.update(1)
                        continue
                    
                    # Create overlay - blend original with mask
                    # Only overlay where mask is not white
                    white_mask = np.all(mask_img == [255, 255, 255], axis=2)
                    overlay = frame.copy()
                    
                    # Apply colored mask with transparency where it's not white
                    alpha = 0.3  # Transparency for mask overlay
                    overlay[~white_mask] = (
                        frame[~white_mask] * (1 - alpha) + 
                        mask_img[~white_mask] * alpha
                    ).astype(np.uint8)
                    
                    # Add frame number (use original frame number, not sequential index)
                    overlay = self.add_frame_number_overlay(overlay, original_frame_number)
                    
                    out.write(overlay)
                    pbar.update(1)
            
            out.release()
            print(f"Visualization video saved to: {output_video}")
    
    def process_all_folders(self, model_path, all_folder_path, downsample_rate=1, batch_size=16):
        """
        Process multiple parent folders, each containing camera-rgb-512x512 subfolder.
        
        Args:
            model_path: Path to trained model
            all_folder_path: Path to folder containing multiple parent folders
            downsample_rate: Process every Nth image (1 = all images, 3 = every 3rd image)
            batch_size: Number of images to process in each batch
            
        Returns:
            List of output folder paths
        """
        # Load model once for all folders
        if model_path:
            self.load_model(model_path)
        
        all_folder = Path(all_folder_path)
        if not all_folder.exists():
            raise ValueError(f"All folder path does not exist: {all_folder_path}")
        
        # Find all subfolders that contain camera-rgb-512x512
        parent_folders = []
        for subfolder in all_folder.iterdir():
            if subfolder.is_dir():
                camera_folder = subfolder / "camera-rgb-512x512"
                if camera_folder.exists():
                    parent_folders.append(subfolder)
        
        if not parent_folders:
            print(f"No valid parent folders with camera-rgb-512x512 found in {all_folder_path}")
            return []
        
        print(f"Found {len(parent_folders)} parent folders to process:")
        for folder in parent_folders:
            print(f"  - {folder.name}")
        
        output_folders = []
        
        # Process each parent folder
        for idx, parent_folder in enumerate(parent_folders, 1):
            print(f"\n{'='*60}")
            print(f"Processing folder {idx}/{len(parent_folders)}: {parent_folder.name}")
            print(f"{'='*60}")
            
            try:
                # Process this parent folder
                output_folder = self.process_image_folder(
                    model_path=None,  # Model already loaded
                    parent_folder=str(parent_folder),
                    downsample_rate=downsample_rate,
                    batch_size=batch_size
                )
                output_folders.append(output_folder)
                print(f"✓ Completed: {parent_folder.name}")
                
            except Exception as e:
                print(f"✗ Error processing {parent_folder.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"All folders processing complete!")
        print(f"Successfully processed {len(output_folders)}/{len(parent_folders)} folders")
        print(f"{'='*60}")
        
        return output_folders


def debug_labelme_conversion(data_root, output_dir, num_samples=5):
    """
    Debug function to check labelme to mask conversion.
    
    Args:
        data_root: Root directory with labelme data
        output_dir: Directory to save debug visualizations
        num_samples: Number of samples to check
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset without transforms to see raw masks
    dataset = LabelmeSegmentationDataset(data_root, transform=None, image_processor=None)
    
    print(f"Checking {min(num_samples, len(dataset))} samples for mask conversion...")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        mask = sample['mask']
        
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask unique values: {np.unique(mask)}")
        
        # Count pixels per class
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        for class_idx, count in zip(unique_classes, counts):
            class_name = ['background', 'arm-left', 'arm-right', 'person'][class_idx] if class_idx < 4 else f"class_{class_idx}"
            percentage = (count / total_pixels) * 100
            print(f"  {class_name}: {count} pixels ({percentage:.1f}%)")
        
        # Save visualization
        fig_path = output_dir / f"debug_sample_{i+1}.png"
        
        # Create colored mask for visualization
        colored_mask = np.zeros_like(image)
        colors = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # Red for left, Green for right, Blue for person

        for class_idx, color in colors.items():
            colored_mask[mask == class_idx] = color
        
        # Create side-by-side visualization
        combined = np.hstack([image, colored_mask])
        cv2.imwrite(str(fig_path), combined)
        print(f"  Debug visualization saved: {fig_path}")
        
        if np.sum(mask > 0) == 0:
            # Check if this is a pure background sample
            png_path, json_path, is_background = dataset.samples[i]
            if is_background:
                print(f"  (This is a pure background sample)")
            else:
                print(f"WARNING: No arm pixels found in sample {i+1}!")
                print(f"  JSON file: {json_path}")
            
            # Load and inspect JSON (only if not a background sample)
            if not is_background and json_path:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    print(f"  JSON shapes found: {len(data.get('shapes', []))}")
                    for shape in data.get('shapes', []):
                        print(f"    - Label: {shape['label']}, Points: {len(shape['points'])}")
                except Exception as e:
                    print(f"  Error reading JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description='SegFormer training and inference for arm segmentation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train SegFormer model')
    train_parser.add_argument('data_root', type=str, help='Root directory with training data')
    train_parser.add_argument('output_dir', type=str, help='Output directory for model checkpoints')
    train_parser.add_argument('--model', type=str, default='mattmdjaga/segformer_b2_clothes',
                             help='Pretrained SegFormer model name')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    train_parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    train_parser.add_argument('--save-every', type=int, default=10, help='Save model every N epochs')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Video inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on video')
    infer_parser.add_argument('model_path', type=str, help='Path to trained model')
    infer_parser.add_argument('input_video', type=str, help='Input video file')
    infer_parser.add_argument('output_video', type=str, help='Output video file')
    infer_parser.add_argument('--fps', type=float, default=None, help='Output FPS')
    infer_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    infer_parser.add_argument('--output-size', type=int, nargs=2, default=[320, 320], help='Output video size (width height)')
    infer_parser.add_argument('--no-batch', action='store_true', help='Disable batch inference')
    infer_parser.add_argument('--no-ffmpeg', action='store_true', help='Use OpenCV instead of ffmpeg for video writing')
    infer_parser.add_argument('--no-overlay', action='store_true', help='Show segmentation only')
    infer_parser.add_argument('--save-frames', type=str, default=None, help='Directory to save individual frames as PNG and labelme JSON files')
    infer_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Image inference command
    image_parser = subparsers.add_parser('image', help='Run inference on single image')
    image_parser.add_argument('model_path', type=str, help='Path to trained model')
    image_parser.add_argument('input_image', type=str, help='Input image file')
    image_parser.add_argument('output_image', type=str, help='Output image file (base name)')
    image_parser.add_argument('--no-overlay', action='store_true', help='Skip overlay generation')
    image_parser.add_argument('--no-mask', action='store_true', help='Skip mask generation')
    image_parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    # Folder inference command
    folder_parser = subparsers.add_parser('folder', help='Run inference on folder of images')
    folder_parser.add_argument('model_path', type=str, help='Path to trained model')
    folder_parser.add_argument('parent_folder', type=str, help='Parent folder containing camera-rgb-512x512 subfolder')
    folder_parser.add_argument('--downsample', type=int, default=3, help='Downsample rate (e.g., 3 = process every 3rd image)')
    folder_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    folder_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # All folders inference command
    all_folder_parser = subparsers.add_parser('all_folder', help='Run inference on multiple folders')
    all_folder_parser.add_argument('model_path', type=str, help='Path to trained model')
    all_folder_parser.add_argument('all_folder_path', type=str, help='Path containing multiple parent folders with camera-rgb-512x512')
    all_folder_parser.add_argument('--downsample', type=int, default=3, help='Downsample rate (e.g., 3 = process every 3rd image)')
    all_folder_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    all_folder_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug labelme to mask conversion')
    debug_parser.add_argument('data_root', type=str, help='Root directory with training data')
    debug_parser.add_argument('output_dir', type=str, help='Output directory for debug visualizations')
    debug_parser.add_argument('--samples', type=int, default=5, help='Number of samples to check')


    torch.cuda.set_device(0) if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Initialize trainer
        trainer = SegFormerTrainer(
            model_name=args.model,
            device=args.device
        )
        
        # Start training
        trainer.train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split,
            save_every=args.save_every
        )
        
    elif args.command == 'infer':
        # Initialize trainer and load model
        trainer = SegFormerTrainer(device=args.device)
        trainer.load_model(args.model_path)
        
        # Process video
        trainer.process_video(
            input_video=args.input_video,
            output_video=args.output_video,
            fps=args.fps,
            show_overlay=not args.no_overlay,
            batch_size=args.batch_size,
            use_batch_inference=not args.no_batch,
            output_size=tuple(args.output_size),
            use_ffmpeg=not args.no_ffmpeg,
            save_frames_dir=args.save_frames
        )
        
    elif args.command == 'image':
        # Initialize trainer and load model
        trainer = SegFormerTrainer(device=args.device)
        trainer.load_model(args.model_path)
        
        # Process single image
        trainer.process_image_file(
            input_image=args.input_image,
            output_image=args.output_image,
            save_overlay=not args.no_overlay,
            save_mask=not args.no_mask
        )
    
    elif args.command == 'folder':
        # Initialize trainer for folder processing
        trainer = SegFormerTrainer(device=args.device)
        
        # Process folder of images
        output_folder = trainer.process_image_folder(
            model_path=args.model_path,
            parent_folder=args.parent_folder,
            downsample_rate=args.downsample,
            batch_size=args.batch_size
        )
        print(f"\nSegmentation masks saved to: {output_folder}")
    
    elif args.command == 'all_folder':
        # Initialize trainer for processing multiple folders
        trainer = SegFormerTrainer(device=args.device)
        
        # Process all folders
        output_folders = trainer.process_all_folders(
            model_path=args.model_path,
            all_folder_path=args.all_folder_path,
            downsample_rate=args.downsample,
            batch_size=args.batch_size
        )
        
        if output_folders:
            print(f"\nAll processing complete! Created {len(output_folders)} output folders:")
            for folder in output_folders:
                print(f"  - {folder}")
        else:
            print("\nNo folders were processed successfully.")
        
    elif args.command == 'debug':
        # Debug labelme to mask conversion
        debug_labelme_conversion(
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_samples=args.samples
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()