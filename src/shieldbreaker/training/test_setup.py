#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from pathlib import Path


def test_imports():
    """Test if required Python packages are installed"""
    print("=" * 60)
    print("Testing Python packages...")
    print("=" * 60)
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "tokenizers": "Tokenizers",
    }
    
    failed = []
    for package, name in required_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"✗ {name:20s} not installed")
            failed.append(package)
    
    if failed:
        print(f"\nMissing dependencies: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True


def test_cuda():

    print("\n" + "=" * 60)
    print("Testing CUDA...")
    print("=" * 60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}:")
                print(f"  Name: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
        else:
            print("⚠️  Warning: CUDA not available, CPU will be used for training (slow)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tokenizer():

    print("\n" + "=" * 60)
    print("Testing Tokenizer...")
    print("=" * 60)
    
    tokenizer_path = Path("datasets/lm_data")
    
    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    
    print(f"Tokenizer path: {tokenizer_path.absolute()}")
    
    if not tokenizer_path.exists():
        print(f"✗ Directory does not exist: {tokenizer_path}")
        return False
    
    missing = []
    for fname in required_files:
        fpath = tokenizer_path / fname
        if fpath.exists():
            print(f"✓ {fname}")
        else:
            print(f"✗ {fname} (missing)")
            missing.append(fname)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    
    # Try loading tokenizer
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        print(f"\n✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Special tokens:")
        print(f"    PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"    BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"    EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"    UNK: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
        
        # Test encoding
        test_text = "<BOS> <PROC:msmpeng.exe> NtOpenKey <EOS>"
        tokens = tokenizer.tokenize(test_text)
        print(f"\n  Test encoding: {test_text}")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        return True
    except Exception as e:
        print(f"\n✗ Tokenizer loading failed: {e}")
        return False


def test_data_files():

    print("\n" + "=" * 60)
    print("Testing data files...")
    print("=" * 60)
    
    data_files = {
        "datasets/lm_data/train.txt": "Training data",
        "datasets/lm_data/valid.txt": "Validation data",
    }
    
    missing = []
    for fpath, desc in data_files.items():
        path = Path(fpath)
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"✓ {desc:20s} {fpath:30s} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {desc:20s} {fpath:30s} (missing)")
            missing.append(fpath)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    
    return True


def test_model_download():

    print("\n" + "=" * 60)
    print("Testing model access...")
    print("=" * 60)
    
    try:
        from transformers import AutoConfig
        
        print("Attempting to access GPT2-XL configuration...")
        config = AutoConfig.from_pretrained("gpt2-xl")
        print(f"✓ HuggingFace accessible")
        print(f"  GPT2-XL configuration:")
        print(f"    Vocabulary size: {config.vocab_size}")
        print(f"    Hidden dimension: {config.n_embd}")
        print(f"    Number of layers: {config.n_layer}")
        print(f"    Attention heads: {config.n_head}")
        
        return True
    except Exception as e:
        print(f"⚠️  Warning: Cannot access HuggingFace ({e})")
        print("   The model will be downloaded on first training, ensure network access.")
        return True  # not fatal


def main():
    print("\n" + "=" * 60)
    print("GPT2-XL Training Environment Test")
    print("=" * 60)
    
    tests = [
        ("Python packages", test_imports),
        ("CUDA", test_cuda),
        ("Tokenizer", test_tokenizer),
        ("Data files", test_data_files),
        ("Model access", test_model_download),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to train.")
        print("\nRun training:")
        print("  Windows: .\\run_train.ps1 or run_train.bat")
        print("  Linux/Mac: ./run_train.sh")
        return 0
    else:
        print("\n✗ Some tests failed, please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
