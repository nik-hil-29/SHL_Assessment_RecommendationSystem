#!/bin/bash
# Install pip packages
pip install -r requirements.txt

# Install Playwright dependencies
pip install playwright
playwright install-deps
playwright install
