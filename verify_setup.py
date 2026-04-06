#!/usr/bin/env python3
"""
Setup and verification script for Warehouse Logistics Environment.

Verifies all components are properly installed and configured.
"""

import os
import sys
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("[FAIL] Python 3.9+ required")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_project_structure():
    """Verify project structure."""
    required_files = [
        'env/__init__.py',
        'env/models.py',
        'env/environment.py',
        'env/tasks.py',
        'env/grader.py',
        'env/utils.py',
        'inference.py',
        'openenv.yaml',
        'requirements.txt',
        'Dockerfile',
        'README.md',
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} (missing)")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pydantic',
        'openai',
        'dotenv',
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            if package == 'dotenv':
                pkg_name = 'dotenv'
            else:
                pkg_name = package
            
            importlib.import_module(pkg_name.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} (not installed)")
            all_ok = False
    
    return all_ok


def check_environment_setup():
    """Check environment variable setup."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_example.exists():
        print(f"[OK] .env.example found")
    else:
        print(f"[FAIL] .env.example (missing)")
        return False
    
    if env_file.exists():
        print(f"[OK] .env configured")
        return True
    else:
        print(f"[WARN] .env not found (run: cp .env.example .env)")
        return False


def test_imports():
    """Test if core modules can be imported."""
    try:
        from env import WarehouseLogisticsEnvironment
        from env.models import Action, ActionType, Observation
        from env.tasks import get_task
        from env.grader import get_grader
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_environment_creation():
    """Test basic environment creation."""
    try:
        from env import WarehouseLogisticsEnvironment
        
        env = WarehouseLogisticsEnvironment(task_difficulty="easy")
        obs = env.reset()
        
        assert len(obs.orders) > 0, "No orders created"
        assert len(obs.warehouses) > 0, "No warehouses created"
        
        print("[OK] Environment creation successful")
        return True
    except Exception as e:
        print(f"[FAIL] Environment creation failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("WAREHOUSE LOGISTICS ENVIRONMENT - SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("Environment Setup", check_environment_setup),
        ("Module Imports", test_imports),
        ("Environment Creation", test_environment_creation),
    ]
    
    def status_icon(result):
        return "[OK]" if result else "[FAIL]"
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {check_name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[OK] Setup verification complete! Environment is ready.")
        print("\nNext steps:")
        print("1. Configure .env with your OpenAI API key")
        print("2. Run: python test_environment.py (to test without API key)")
        print("3. Run: python inference.py (to test with OpenAI)")
        return 0
    else:
        print("\n[WARN] Some checks failed. Please address issues above.")
        print("\nTo install dependencies:")
        print("pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
