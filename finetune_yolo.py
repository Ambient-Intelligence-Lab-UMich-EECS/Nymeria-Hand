import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml
from ultralytics import YOLO
import argparse
from tqdm import tqdm
from pqdm.processes import pqdm
import cv2


def convert_labelme_to_yolo(json_path, img_width, img_height, include_background=True):
    """Convert labelme JSON format to YOLO format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []
    
    # Map class names to indices
    if include_background:
        class_map = {'left': 0, 'right': 1, 'background': 2}
    else:
        class_map = {'left': 0, 'right': 1}
    
    for shape in data['shapes']:
        if shape['shape_type'] != 'rectangle':
            continue
            
        label = shape['label']
        if label not in class_map:
            continue
            
        points = shape['points']
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Convert to YOLO format (normalized center x, y, width, height)
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        yolo_annotations.append(f"{class_map[label]} {center_x} {center_y} {width} {height}")
    
    return yolo_annotations


def prepare_yolo_dataset(source_dir, output_dir, train_ratio=0.8, include_background=True):
    """Prepare dataset in YOLO format."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Collect all annotated images
    annotated_images = []
    
    for subfolder in source_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        for json_file in subfolder.glob('*.json'):
            png_file = json_file.with_suffix('.png')
            if png_file.exists():
                annotated_images.append((json_file, png_file))
    
    print(f"Found {len(annotated_images)} annotated images")
    
    # Shuffle and split dataset
    random.shuffle(annotated_images)
    split_idx = int(len(annotated_images) * train_ratio)
    train_set = annotated_images[:split_idx]
    val_set = annotated_images[split_idx:]
    
    print(f"Train set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")
    
    # Process train set
    for json_path, img_path in tqdm(train_set, desc="Processing train set"):
        process_image(json_path, img_path, output_path, 'train', include_background)
    
    # Process validation set
    for json_path, img_path in tqdm(val_set, desc="Processing validation set"):
        process_image(json_path, img_path, output_path, 'val', include_background)
    
    # Create dataset.yaml
    if include_background:
        dataset_config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'left',
                1: 'right',
                2: 'background'
            },
            'nc': 3
        }
    else:
        dataset_config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'left',
                1: 'right'
            },
            'nc': 2
        }
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset prepared at {output_path}")
    return output_path / 'dataset.yaml'


def process_image(json_path, img_path, output_path, split, include_background=True):
    """Process a single image and its annotation."""
    # Get image dimensions
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # Convert annotations
    yolo_annotations = convert_labelme_to_yolo(json_path, img_width, img_height, include_background)
    
    if not yolo_annotations:
        return
    
    # Copy image
    img_name = img_path.name
    shutil.copy(img_path, output_path / 'images' / split / img_name)
    
    # Save labels
    label_name = img_path.stem + '.txt'
    label_path = output_path / 'labels' / split / label_name
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))


def train_yolo(data_yaml, model_name='yolov11x.pt', epochs=100, imgsz=640, batch=16):
    """Train YOLO model on the prepared dataset."""
    # Load a model
    model = YOLO(model_name)
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='hand_detection',
        patience=40,
        save=True,
        device='cuda',
        workers=8,
        project='runs/train',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,

        hsv_v=0.8,
        hsv_h=0.02,
        degrees=15.0,
        mosaic=0.4,
        scale=0.4,
        translate=0.2,
        shear=5.0,

        plots=True,
        val=True
    )
    
    return model


def validate_model(model, data_yaml):
    """Validate the trained model."""
    metrics = model.val(data=data_yaml)
    
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def process_inference_image(args):
    """Process a single image for inference - separated for pqdm compatibility."""
    img_path, boxes_data, output_path, class_names, colors, font, filter_background = args
    
    # Load image only when needed
    img = Image.open(img_path)
    
    # Create a copy of the image for drawing
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Draw bounding boxes from serialized data
    if boxes_data:
        for box_data in boxes_data:
            x1, y1, x2, y2, conf, cls = box_data
            
            # Skip background predictions if filtering is enabled
            if filter_background and cls == 2:
                continue
            
            # Draw bounding box
            color = colors.get(cls, (255, 255, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_names.get(cls, 'unknown')} {conf:.2f}"
            
            # Get text size for background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background for text
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], 
                         fill=color, outline=color)
            
            # Draw text
            draw.text((x1 + 5, y1 - text_height - 2), label, fill=(0, 0, 0), font=font)
    
    # Save annotated image
    output_img_path = output_path / img_path.name
    img_copy.save(output_img_path)
    
    # Close the image immediately to free memory
    img.close()
    del img_copy
    
    return str(img_path)


def run_inference(model_path, input_dir, output_dir, conf_threshold=0.5, batch_size=32, filter_background=True):
    """Run inference on images using trained YOLO model with batched processing."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the trained model
    model = YOLO(model_path)
    model.to('cuda')  # Move model to CUDA device
    
    # Class names - only show left/right in output
    class_names = {0: 'left', 1: 'right', 2: 'background'}
    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (128, 128, 128)}  # Red for left, Green for right, Gray for background
    
    # Find all PNG images and sort them numerically
    png_files = sorted(list(input_path.glob('*.png')), key=lambda x: x.name)
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files. Processing in batches of {batch_size}...")
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Process images in batches
    total_processed = 0
    num_batches = (len(png_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(png_files))
        batch_files = png_files[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} images)...")
        
        # Convert paths to strings for YOLO (no need to load images first)
        batch_paths = batch_files
        path_strings = [str(p) for p in batch_paths]
        
        # Run batch inference directly on file paths
        print(f"Running inference on batch {batch_idx + 1}...")
        results = model(path_strings, conf=conf_threshold)
        
        # Extract tensor data to CPU for parallel processing (avoid CUDA pickling issues)
        process_args = []
        for img_path, result in zip(batch_paths, results):
            # Extract box data from CUDA tensors to CPU numpy arrays
            boxes_data = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    boxes_data.append((x1, y1, x2, y2, conf, cls))
            
            process_args.append((img_path, boxes_data, output_path, class_names, colors, font, filter_background))
        
        # Use pqdm with 6 workers
        pqdm(process_args, process_inference_image, n_jobs=12, desc=f"Processing batch {batch_idx + 1}")
            
        total_processed += len(batch_files)
        print(f"Batch {batch_idx + 1} complete. Total processed: {total_processed}/{len(png_files)}")
        
        # Clear batch results to free memory
        del results
    
    print(f"\nInference complete! All {total_processed} annotated images saved to {output_dir}")


def run_video_inference(model_path, input_video, output_video, conf_threshold=0.5, filter_background=True, fps=None):
    """Run inference on video using trained YOLO model and generate annotated video."""
    # Load the trained model
    model = YOLO(model_path)
    model.to('cuda')  # Move model to CUDA device
    
    # Class names
    class_names = {0: 'left', 1: 'right', 2: 'background'}
    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (128, 128, 128)}  # Red for left, Green for right, Gray for background
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use original FPS if not specified
    if fps is None:
        fps = original_fps
    
    print(f"Input video: {input_video}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {original_fps} (output: {fps})")
    print(f"Total frames: {total_frames}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"Error: Could not open output video {output_video}")
        cap.release()
        return
    
    # Process video frame by frame
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO inference on the frame
            results = model(frame, conf=conf_threshold)
            
            # Draw bounding boxes on the frame
            annotated_frame = frame.copy()
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Skip background predictions if filtering is enabled
                        if filter_background and cls == 2:
                            continue
                        
                        # Get color for the class
                        color = colors.get(cls, (255, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Prepare label
                        label = f"{class_names.get(cls, 'unknown')} {conf:.2f}"
                        
                        # Calculate text size
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            color,
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0),
                            2
                        )
            
            # Write the annotated frame
            out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\nVideo inference complete! Processed {frame_count} frames.")
    print(f"Annotated video saved to: {output_video}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO for hand detection')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'video'], default='train',
                        help='Mode: train, inference (for images), or video')
    parser.add_argument('--source', type=str, default='dataset/nymeria_hand',
                        help='Path to source dataset directory (training), input images directory (inference), or input video file (video mode)')
    parser.add_argument('--output', type=str, default='dataset/yolo_hands',
                        help='Path to output YOLO dataset directory (training), output images directory (inference), or output video file (video mode)')
    parser.add_argument('--model', type=str, default='yolo11l.pt',
                        help='Base YOLO model to use for training or path to trained model for inference/video')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Only prepare dataset without training')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for inference')
    parser.add_argument('--inference-batch-size', type=int, default=64,
                        help='Batch size for inference mode')
    parser.add_argument('--no-background', action='store_true',
                        help='Exclude background class during training (default: include background)')
    parser.add_argument('--show-background', action='store_true',
                        help='Show background predictions during inference (default: filter out background)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS for video mode (default: use original FPS)')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        # Run video inference mode
        print("Running video inference...")
        filter_bg = not args.show_background if hasattr(args, 'show_background') else True
        run_video_inference(args.model, args.source, args.output, args.conf, filter_bg, args.fps)
    elif args.mode == 'inference':
        # Run inference mode
        print("Running inference...")
        filter_bg = not args.show_background if hasattr(args, 'show_background') else True
        run_inference(args.model, args.source, args.output, args.conf, args.inference_batch_size, filter_bg)
    else:
        # Training mode
        # Prepare dataset
        print("Preparing YOLO dataset...")
        include_bg = not args.no_background if hasattr(args, 'no_background') else True
        data_yaml = prepare_yolo_dataset(
            args.source,
            args.output,
            train_ratio=args.train_ratio,
            include_background=include_bg
        )
        
        if args.prepare_only:
            print("Dataset preparation complete. Exiting.")
            return
        
        # Train model
        print("\nStarting YOLO training...")
        model = train_yolo(
            data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        # Validate model
        print("\nValidating model...")
        validate_model(model, data_yaml)
        
        # Export model
        print("\nExporting model...")
        
        print("\nTraining complete! Model saved in runs/train/hand_detection/")
        print("Best weights: runs/train/hand_detection/weights/best.pt")
        print("Last weights: runs/train/hand_detection/weights/last.pt")


if __name__ == "__main__":
    main()