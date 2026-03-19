# RakutenAI-3.0 vs DeepSeek-V3: Technical Investigation Report

## 📋 Overview

This repository contains a comprehensive technical investigation comparing RakutenAI-3.0 and DeepSeek-V3 models, including quantitative analysis of 10,929 tensors across 61 layers.

## 🔍 Key Findings

- **99.94% average cosine similarity** between model weights
- **85.82% of tensors** show extremely high similarity (>0.999)
- **LoRA implementation confirmed**: Q/KV projection modifications detected
- **Identical tokenizer**: SHA256 hash verification confirms same source
- **Architecture match**: Complete parameter alignment verified

## 📁 Repository Structure

```
RakutenAI_Report/
├── README.md                    # This file
├── docs/                        # Documentation
│   ├── README.md               # Full technical report (Japanese)
│   └── verification_guide.md   # Step-by-step verification guide
├── scripts/                     # Analysis scripts
│   ├── comprehensive_model_analysis.py  # Main analysis tool
│   └── lora_parameter_analysis.py      # LoRA-specific analysis
├── data/                        # Analysis results
│   └── comprehensive_analysis_results.csv  # Detailed tensor comparison data
├── images/                      # Visualizations
│   └── weight_similarity_analysis.png     # Similarity distribution plots
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Verification
```python
# Verify tokenizer identity
import hashlib
from huggingface_hub import hf_hub_download

file_a = hf_hub_download('deepseek-ai/DeepSeek-V3', 'tokenizer.json')
file_b = hf_hub_download('Rakuten/RakutenAI-3.0', 'tokenizer.json')

hash_a = hashlib.sha256(open(file_a, 'rb').read()).hexdigest()
hash_b = hashlib.sha256(open(file_b, 'rb').read()).hexdigest()

print(f"DeepSeek-V3: {hash_a}")
print(f"RakutenAI:   {hash_b}")
print(f"Match: {hash_a == hash_b}")  # Returns: True
```

### LoRA Parameter Detection
```python
# Check for LoRA parameters
from safetensors.torch import load_file

file = hf_hub_download('Rakuten/RakutenAI-3.0', 'model-00001-of-000163.safetensors')
weights = load_file(file, device='cpu')

lora_keys = [k for k in weights.keys() if '_a_proj' in k or '_b_proj' in k]
print(f"Found {len(lora_keys)} LoRA parameters")
```

### Full Analysis
```bash
# Run comprehensive comparison
python scripts/comprehensive_model_analysis.py \
    --model-a "deepseek-ai/DeepSeek-V3" \
    --model-b "Rakuten/RakutenAI-3.0" \
    --num-files 20 \
    --output-dir results/
```

## 📊 Analysis Results

### Tensor-Level Similarity
- **Total tensors analyzed**: 10,929
- **Files processed**: 20 (model-00001 through model-00020)
- **Layers covered**: 61 (Layer 0-60)

### Component-Wise Analysis
| Component | Average Similarity | Notes |
|-----------|-------------------|-------|
| MLP layers | 99.9%+ | Virtually identical |
| LayerNorm | 100% | Perfect match |
| Attention (non-LoRA) | 99.8%+ | Minimal differences |
| Attention (LoRA) | 97-98% | LoRA modifications detected |

## 🔬 Technical Details

### LoRA Configuration Detected
```
Q (Query) LoRA:
- A matrix: [1536, 7168] 
- B matrix: [24576, 1536]
- Rank: 1536

KV (Key/Value) LoRA:  
- A matrix: [576, 7168]
- B matrix: [32768, 512] 
- Rank: 512

Strategic 3:1 rank allocation (Q:KV = 1536:512)
```

### Architecture Verification
- **Model type**: Both show `"model_type": "deepseek_v3"`
- **Hidden size**: 7168 (identical)
- **Layers**: 61 (identical)
- **Attention heads**: Complete parameter match
- **Vocabulary size**: Identical tokenizer confirmed

## 📄 Full Report

The complete technical investigation report (in Japanese) is available in [`docs/README.md`](docs/README.md).

Key sections include:
1. Investigation methodology
2. Quantitative findings
3. LoRA technical analysis
4. Public funding implications (GENIAC project)
5. License compliance assessment
6. Technical evaluation

## 🛠️ Reproducibility

All analysis code and data are provided for independent verification:

- **Scripts**: Complete Python analysis tools
- **Data**: Raw tensor comparison results (CSV format)
- **Documentation**: Step-by-step reproduction guide
- **Visualizations**: Similarity distribution plots

## ⚖️ Legal and Ethical Considerations

This investigation was conducted for academic and transparency purposes, using publicly available models and adhering to their respective licenses.

## 🤝 Contributing

Issues and discussions are welcome. Please ensure any contributions maintain the objective, technical focus of this investigation.

## 📈 Citation

If you use this analysis in your research, please cite:
```bibtex
@misc{rakutenai_investigation_2025,
  title={RakutenAI-3.0 vs DeepSeek-V3: Technical Investigation Report},
  year={2025},
  url={https://github.com/hama-jp/RakutenAI_Report}
}
```

---

**Note**: This investigation focuses on technical analysis and transparency in AI model development. All findings are presented objectively for the benefit of the AI research community.