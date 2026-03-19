#!/usr/bin/env python3
"""
GPU-Accelerated Model Audit Script
Maximum performance version using GPU acceleration and parallel processing
"""

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json
from collections import defaultdict
from tqdm import tqdm
import logging
import concurrent.futures
import threading
from queue import Queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUAcceleratedAuditor:
    def __init__(self):
        self.model_a = 'deepseek-ai/DeepSeek-V3'
        self.model_b = 'Rakuten/RakutenAI-3.0'
        self.cache_dir = '/mnt/d/huggingface_cache'
        self.all_layer_data = {}
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Set high memory usage
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Threading setup
        self.download_queue = Queue()
        self.analysis_queue = Queue()
        self.results_lock = threading.Lock()
    
    def download_file_pair(self, file_idx):
        """Download both model files for a given index"""
        filename = f'model-{file_idx:05d}-of-000163.safetensors'
        
        try:
            print(f"[Download {file_idx}] Starting {filename}")
            
            # Download in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(
                    hf_hub_download, self.model_a, filename, cache_dir=self.cache_dir
                )
                future_b = executor.submit(
                    hf_hub_download, self.model_b, filename, cache_dir=self.cache_dir
                )
                
                file_a = future_a.result()
                file_b = future_b.result()
            
            print(f"[Download {file_idx}] Complete")
            return file_a, file_b
            
        except Exception as e:
            print(f"[Download {file_idx}] ERROR: {e}")
            return None, None
    
    def gpu_tensor_similarity(self, tensor_a, tensor_b):
        """GPU-accelerated tensor similarity calculation"""
        try:
            # Move to GPU
            a_gpu = tensor_a.float().to(self.device)
            b_gpu = tensor_b.float().to(self.device)
            
            # Flatten on GPU
            a_flat = a_gpu.flatten()
            b_flat = b_gpu.flatten()
            
            # Calculate similarity on GPU
            norm_a = torch.norm(a_flat)
            norm_b = torch.norm(b_flat)
            
            if norm_a > 0 and norm_b > 0:
                similarity = (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).cpu().item()
                return similarity
            return 0.0
            
        except Exception as e:
            # Fallback to CPU
            return self.cpu_tensor_similarity(tensor_a, tensor_b)
    
    def cpu_tensor_similarity(self, tensor_a, tensor_b):
        """CPU fallback for tensor similarity"""
        a_flat = tensor_a.float().flatten()
        b_flat = tensor_b.float().flatten()
        
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        
        if norm_a > 0 and norm_b > 0:
            return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()
        return 0.0
    
    def analyze_file_pair(self, file_a, file_b, file_idx):
        """Analyze a pair of weight files with GPU acceleration"""
        print(f"[Analysis {file_idx}] Loading weights...")
        
        try:
            # Load weights
            weights_a = load_file(file_a, device='cpu')  # Load to CPU first
            weights_b = load_file(file_b, device='cpu')
            
            # Find common tensors
            keys_a = set(weights_a.keys())
            keys_b = set(weights_b.keys())
            common_keys = keys_a & keys_b
            
            print(f"[Analysis {file_idx}] Processing {len(common_keys)} tensors on GPU...")
            
            # Process tensors in batches for GPU efficiency
            batch_results = []
            
            for key in tqdm(common_keys, desc=f"File {file_idx}"):
                # Extract layer number
                match = re.search(r'layers\.(\d+)', key)
                if not match:
                    continue
                
                layer_num = int(match.group(1))
                
                tensor_a = weights_a[key]
                tensor_b = weights_b[key]
                
                if tensor_a.shape != tensor_b.shape:
                    continue
                
                # GPU-accelerated similarity calculation
                similarity = self.gpu_tensor_similarity(tensor_a, tensor_b)
                
                # Classify tensor type
                tensor_type = self.classify_tensor(key)
                
                batch_results.append({
                    'layer': layer_num,
                    'key': key,
                    'similarity': similarity,
                    'tensor_type': tensor_type,
                    'file_idx': file_idx
                })
            
            # Store results thread-safely
            with self.results_lock:
                for result in batch_results:
                    layer_num = result['layer']
                    
                    if layer_num not in self.all_layer_data:
                        self.all_layer_data[layer_num] = {
                            'tensors': [],
                            'similarities': [],
                            'files_seen': set(),
                            'tensor_types': defaultdict(list)
                        }
                    
                    self.all_layer_data[layer_num]['tensors'].append(result['key'])
                    self.all_layer_data[layer_num]['similarities'].append(result['similarity'])
                    self.all_layer_data[layer_num]['files_seen'].add(result['file_idx'])
                    self.all_layer_data[layer_num]['tensor_types'][result['tensor_type']].append(result['similarity'])
            
            print(f"[Analysis {file_idx}] Complete - {len(batch_results)} tensors processed")
            
            # Clean up files
            Path(file_a).unlink(missing_ok=True)
            Path(file_b).unlink(missing_ok=True)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Analysis {file_idx}] ERROR: {e}")
    
    def classify_tensor(self, name):
        """Classify tensor by type"""
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            return 'attention'
        elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
            return 'mlp'
        elif 'norm' in name:
            return 'norm'
        elif 'embed' in name:
            return 'embedding'
        else:
            return 'other'
    
    def parallel_audit(self, num_files=20):
        """Run audit with maximum parallelization"""
        print('='*80)
        print('GPU-ACCELERATED MODEL AUDIT - MAXIMUM PERFORMANCE')
        print('='*80)
        print(f'Model A: {self.model_a}')
        print(f'Model B: {self.model_b}')
        print(f'Files to analyze: {num_files}')
        print(f'Device: {self.device}')
        print('='*80)
        
        file_indices = list(range(1, num_files + 1))
        
        # Process files with threading for maximum throughput
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for file_idx in file_indices:
                future = executor.submit(self.process_single_file, file_idx)
                futures.append(future)
            
            # Wait for completion with progress
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    future.result()
                    print(f"Progress: {i+1}/{num_files} files complete")
                except Exception as e:
                    print(f"File processing error: {e}")
        
        return self.generate_comprehensive_report()
    
    def process_single_file(self, file_idx):
        """Process a single file index"""
        file_a, file_b = self.download_file_pair(file_idx)
        if file_a and file_b:
            self.analyze_file_pair(file_a, file_b, file_idx)
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        print('\n' + '='*80)
        print('GPU-ACCELERATED AUDIT RESULTS')
        print('='*80)
        
        # Prepare summary data
        layer_summaries = []
        
        for layer_num in sorted(self.all_layer_data.keys()):
            data = self.all_layer_data[layer_num]
            
            if data['similarities']:
                sims = data['similarities']
                
                summary = {
                    'layer': layer_num,
                    'num_tensors': len(data['tensors']),
                    'num_files': len(data['files_seen']),
                    'avg_cosine': np.mean(sims),
                    'min_cosine': np.min(sims),
                    'max_cosine': np.max(sims),
                    'std_cosine': np.std(sims),
                    'median_cosine': np.median(sims),
                    'pct_over_999': sum(1 for s in sims if s > 0.999) / len(sims) * 100,
                    'pct_over_99': sum(1 for s in sims if s > 0.99) / len(sims) * 100,
                    'pct_under_99': sum(1 for s in sims if s <= 0.99) / len(sims) * 100
                }
                
                # Add tensor type breakdown
                for ttype, tsims in data['tensor_types'].items():
                    if tsims:
                        summary[f'{ttype}_avg'] = np.mean(tsims)
                
                layer_summaries.append(summary)
        
        # Create DataFrame
        df = pd.DataFrame(layer_summaries)
        
        # Save to CSV
        df.to_csv('gpu_accelerated_audit_results.csv', index=False)
        print(f'Saved detailed results to: gpu_accelerated_audit_results.csv')
        
        # Print summary
        if len(df) > 0:
            all_sims = []
            for data in self.all_layer_data.values():
                all_sims.extend(data['similarities'])
            
            print(f'\nOVERALL AUDIT SUMMARY:')
            print(f'Total layers analyzed: {len(df)}')
            print(f'Total tensors compared: {len(all_sims):,}')
            print(f'Overall average cosine similarity: {np.mean(all_sims):.8f}')
            print(f'Tensors with similarity > 0.999: {sum(1 for s in all_sims if s > 0.999):,} ({sum(1 for s in all_sims if s > 0.999)/len(all_sims)*100:.2f}%)')
            print(f'Tensors with similarity > 0.99:  {sum(1 for s in all_sims if s > 0.99):,} ({sum(1 for s in all_sims if s > 0.99)/len(all_sims)*100:.2f}%)')
        
        return df

# Run the GPU-accelerated audit
if __name__ == "__main__":
    auditor = GPUAcceleratedAuditor()
    results = auditor.parallel_audit(num_files=20)  # Process 20 files for comprehensive coverage
    
    print('\n' + '='*80)
    print('GPU-ACCELERATED AUDIT COMPLETE')
    print('='*80)