"""
Simple per-class replay buffer for incremental/continual learning.
Stores metadata (image paths, labels, etc.) and samples uniformly from all stored classes.
"""

import os
import json
import random
import logging

logger = logging.getLogger("global_logger")


class ReplayBuffer:
    """
    A simple per-class replay buffer that stores sample metadata.
    
    During training, a subset of old-class data is stored. When new classes
    are added incrementally, replay samples are mixed into training batches
    to prevent catastrophic forgetting.
    
    Storage is lightweight: only metadata dicts (filenames, labels, etc.)
    are stored, not raw tensors.
    """
    
    def __init__(self, max_per_class=50):
        """
        Args:
            max_per_class: Maximum number of samples to store per class
        """
        self.max_per_class = max_per_class
        self.buffers = {}  # class_name -> list of sample metadata dicts
    
    def add_batch(self, batch_data):
        """
        Add samples from a training batch to the buffer.
        
        Args:
            batch_data: dict with keys like 'filename', 'clsname', 'label', etc.
                       Values are lists (one element per sample in batch).
        """
        batch_size = len(batch_data["filename"])
        
        for i in range(batch_size):
            cls_name = batch_data["clsname"][i]
            
            if cls_name not in self.buffers:
                self.buffers[cls_name] = []
            
            sample = {
                "filename": batch_data["filename"][i],
                "label": batch_data["label"][i] if isinstance(batch_data["label"], list) else batch_data["label"][i].item(),
                "clsname": cls_name,
            }
            
            # Add maskname if available
            if "maskname" in batch_data and batch_data["maskname"] is not None:
                sample["maskname"] = batch_data["maskname"][i]
            
            # Reservoir sampling: if buffer is full, replace randomly
            if len(self.buffers[cls_name]) < self.max_per_class:
                self.buffers[cls_name].append(sample)
            else:
                # Random replacement with decreasing probability
                idx = random.randint(0, len(self.buffers[cls_name]))
                if idx < self.max_per_class:
                    self.buffers[cls_name][idx] = sample
    
    def sample(self, num_samples):
        """
        Sample uniformly from all stored classes.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of sample metadata dicts, or empty list if buffer is empty
        """
        all_samples = []
        for cls_samples in self.buffers.values():
            all_samples.extend(cls_samples)
        
        if not all_samples:
            return []
        
        num_samples = min(num_samples, len(all_samples))
        return random.sample(all_samples, num_samples)
    
    def sample_per_class(self, num_per_class):
        """
        Sample a fixed number of samples from each stored class.
        
        Args:
            num_per_class: Number of samples per class
            
        Returns:
            List of sample metadata dicts
        """
        samples = []
        for cls_name, cls_samples in self.buffers.items():
            n = min(num_per_class, len(cls_samples))
            samples.extend(random.sample(cls_samples, n))
        return samples
    
    def get_all_as_json_lines(self):
        """
        Get all stored samples as JSON lines (compatible with dataset meta_file format).
        
        Returns:
            List of JSON-serializable dicts
        """
        all_samples = []
        for cls_samples in self.buffers.values():
            all_samples.extend(cls_samples)
        return all_samples
    
    def num_classes(self):
        """Return number of classes currently in the buffer."""
        return len(self.buffers)
    
    def num_samples(self):
        """Return total number of samples in the buffer."""
        return sum(len(v) for v in self.buffers.values())
    
    def save(self, path):
        """
        Save buffer contents to a JSON file.
        
        Args:
            path: File path to save to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "max_per_class": self.max_per_class,
            "buffers": self.buffers,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"ReplayBuffer saved to {path}: "
                    f"{self.num_classes()} classes, {self.num_samples()} samples")
    
    def load(self, path):
        """
        Load buffer contents from a JSON file.
        
        Args:
            path: File path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"ReplayBuffer file not found: {path}")
            return
        
        with open(path, "r") as f:
            data = json.load(f)
        
        self.max_per_class = data.get("max_per_class", self.max_per_class)
        self.buffers = data.get("buffers", {})
        logger.info(f"ReplayBuffer loaded from {path}: "
                    f"{self.num_classes()} classes, {self.num_samples()} samples")
    
    def __repr__(self):
        cls_info = {k: len(v) for k, v in self.buffers.items()}
        return (f"ReplayBuffer(max_per_class={self.max_per_class}, "
                f"classes={cls_info})")
