#!/bin/bash
# NASA ML Project - Quick Setup Script
# This script sets up the Python environment and installs all dependencies

echo "========================================================================"
echo "NASA ML PROJECT - SETUP SCRIPT"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Project directory: $SCRIPT_DIR"
echo ""

# Check Python version
echo "Checking Python version..."
echo "------------------------------------------------------------------------"
PYTHON_VERSION=$(python3 --version 2>&1)
echo "✓ $PYTHON_VERSION"
echo ""

# Ask user about virtual environment
echo "Setup Options:"
echo "------------------------------------------------------------------------"
echo "1. Install in virtual environment (recommended)"
echo "2. Install globally with --break-system-packages (not recommended)"
echo "3. Cancel setup"
echo ""
read -p "Choose option (1-3): " option

case $option in
    1)
        echo ""
        echo "Setting up virtual environment..."
        echo "------------------------------------------------------------------------"
        
        # Create virtual environment
        python3 -m venv venv
        
        if [ $? -eq 0 ]; then
            echo "✓ Virtual environment created"
        else
            echo "✗ Failed to create virtual environment"
            exit 1
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        if [ $? -eq 0 ]; then
            echo "✓ Virtual environment activated"
        else
            echo "✗ Failed to activate virtual environment"
            exit 1
        fi
        
        # Upgrade pip
        echo ""
        echo "Upgrading pip..."
        pip install --upgrade pip
        
        # Install requirements
        echo ""
        echo "Installing packages from requirements.txt..."
        echo "------------------------------------------------------------------------"
        pip install -r requirements.txt
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Installation successful!"
            echo ""
            echo "To use the project:"
            echo "  1. Activate the virtual environment:"
            echo "     source venv/bin/activate"
            echo ""
            echo "  2. Run the test script:"
            echo "     python test_project.py"
            echo ""
            echo "  3. Run inference:"
            echo "     python local_inference.py"
        else
            echo ""
            echo "✗ Installation failed"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo "Installing globally..."
        echo "------------------------------------------------------------------------"
        echo "⚠ WARNING: This will install packages globally with --break-system-packages"
        read -p "Are you sure? (yes/no): " confirm
        
        if [ "$confirm" = "yes" ]; then
            pip3 install -r requirements.txt --break-system-packages
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ Installation successful!"
                echo ""
                echo "Run test script: python3 test_project.py"
                echo "Run inference: python3 local_inference.py"
            else
                echo ""
                echo "✗ Installation failed"
                exit 1
            fi
        else
            echo "Installation cancelled"
            exit 0
        fi
        ;;
        
    3)
        echo "Setup cancelled"
        exit 0
        ;;
        
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

# Run test if installation was successful
echo ""
echo "========================================================================"
echo "RUNNING TESTS"
echo "========================================================================"
echo ""

if [ $option -eq 1 ]; then
    python test_project.py
else
    python3 test_project.py
fi

echo ""
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  • Review PROJECT_STATUS.md for detailed information"
echo "  • Run: python local_inference.py (to make predictions)"
echo "  • Run: python main_pipeline.py --help (to see data collection options)"
echo ""
