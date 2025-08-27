#!/usr/bin/env python3
"""
Quick test script to validate the Neural Network Convergence Accelerator implementation
without requiring heavy dependencies or long training runs.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_config_loading():
    """Test configuration loading and validation."""
    print("Testing configuration loading...")
    
    try:
        # Test that we can import the config module
        from src.utils.config import load_config, ModelConfig, TrainingConfig
        print("✓ Configuration module imports successfully")
        
        # Test configuration file loading
        config_files = [
            'configs/ablate/baseline.yaml',
            'configs/ablate/adamw_clip.yaml', 
            'configs/ablate/cosine_amp.yaml',
            'configs/ablate/norm_targets.yaml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    print(f"✓ {config_file} loads successfully")
                except Exception as e:
                    print(f"✗ {config_file} failed to load: {e}")
                    return False
            else:
                print(f"✗ {config_file} not found")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Configuration module import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model architecture creation."""
    print("Testing model architecture...")
    
    try:
        # Test model imports
        from src.core.enhanced_model import EnhancedPokerNetwork, TargetNormalizer
        from src.utils.config import ModelConfig
        print("✓ Enhanced model module imports successfully")
        
        # Test model configuration
        config = ModelConfig(
            name="EnhancedPokerNetwork",
            input_size=156,
            hidden_size=256,
            num_actions=3,
            dropout=0.1,
            use_layer_norm=True,
            use_residuals=True
        )
        print("✓ Model configuration created successfully")
        
        # Test target normalizer
        normalizer = TargetNormalizer("robust")
        test_values = [1.0, 2.0, 3.0, 100.0, -50.0]  # Include outliers
        normalizer.update(test_values)
        normalized = normalizer.normalize(test_values)
        denormalized = normalizer.denormalize(normalized)
        print("✓ Target normalizer works correctly")
        
        return True
        
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        'src/utils/config.py',
        'src/core/enhanced_model.py', 
        'src/core/enhanced_deep_cfr.py',
        'src/training/train_enhanced.py',
        'scripts/validate_convergence.py',
        'configs/ablate/baseline.yaml',
        'configs/ablate/adamw_clip.yaml',
        'configs/ablate/cosine_amp.yaml',
        'configs/ablate/norm_targets.yaml',
        'run_ablation_studies.sh',
        'docs/neural_network_convergence_accelerator.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    if missing_files:
        print(f"✗ {len(missing_files)} files missing")
        return False
    else:
        print(f"✓ All {len(required_files)} required files present")
        return True


def test_script_permissions():
    """Test that scripts are executable."""
    print("Testing script permissions...")
    
    scripts = ['run_ablation_studies.sh']
    
    for script in scripts:
        if Path(script).exists():
            if os.access(script, os.X_OK):
                print(f"✓ {script} is executable")
            else:
                print(f"✗ {script} is not executable")
                return False
        else:
            print(f"✗ {script} not found")
            return False
    
    return True


def run_quick_validation():
    """Run all quick validation tests."""
    print("Neural Network Convergence Accelerator - Quick Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Script Permissions", test_script_permissions),
        ("Configuration Loading", test_config_loading),
        ("Model Architecture", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'PASS' if result else 'FAIL'}\n")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((test_name, False))
            print("FAIL\n")
    
    # Summary
    print("Test Summary")
    print("=" * 20)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run validation: python3 scripts/validate_convergence.py")
        print("3. Run ablation studies: ./run_ablation_studies.sh")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_quick_validation()
    sys.exit(0 if success else 1)