# ML Utils

Machine learning utilities and setup scripts for different cloud platforms.

## Features

- Google Cloud Platform setup and initialization
- SSP Cloud (Statistics Service Platform) configuration
- Environment setup scripts for ML workflows
- Cloud-specific utility functions

## Structure

- `google-cloud/` - Google Cloud Platform utilities
  - `init.sh` - GCP initialization script
- `sspcloud/` - SSP Cloud platform utilities  
  - `init.sh` - SSP Cloud initialization script
  - `install.sh` - SSP Cloud installation script

## Usage

### Google Cloud Setup
```bash
cd google-cloud
chmod +x init.sh
./init.sh
```

### SSP Cloud Setup
```bash
cd sspcloud
chmod +x install.sh install.sh
./install.sh
chmod +x init.sh
./init.sh
```

## Notes

This directory contains shell scripts and utilities rather than Python packages, so no Python dependencies are required.