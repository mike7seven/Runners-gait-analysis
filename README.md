# Runners Gait Analysis Project

A comprehensive system for analyzing runners' gait patterns using computer vision and machine learning techniques. This project helps athletes and coaches understand and improve running form through automated video analysis.

## Overview

This repository contains a complete gait analysis system that:

- Processes video footage of runners
- Extracts pose data and movement patterns
- Analyzes gait characteristics
- Provides insights for form improvement
- Offers a REST API for integration with web and mobile applications

## Repository Structure

```text
.
├── docs/                   # Project documentation
├── gait_analysis_app/     # Main application code
│   ├── src/              # Source code modules
│   ├── notebooks/        # Analysis notebooks
│   ├── config/          # Configuration files
│   ├── data/            # Data storage
│   ├── api.py           # REST API implementation
│   ├── API_README.md    # API documentation
│   └── README.md        # Application-specific documentation
└── LICENSE              # Project license
```

## Getting Started

For detailed setup and usage instructions, please refer to the [application documentation](gait_analysis_app/README.md).

### Running the API

To start the REST API server:

1. Install dependencies:
   ```bash
   cd gait_analysis_app
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```bash
   python api.py
   ```

3. Access the API documentation at http://localhost:8000/docs

## Documentation

- [Application Guide](gait_analysis_app/README.md) - Detailed application setup and usage
- [API Documentation](gait_analysis_app/API_README.md) - REST API endpoints and usage
- [Technical Documentation](docs/) - System architecture and technical details

## Web Dashboard

The system architecture includes a Web Dashboard for visualizing gait analysis results. The API is designed to support integration with a React frontend that you can develop separately.

## License

This project is licensed under the terms of the license file in the root directory.

## Contributing

We welcome contributions! Please read our contribution guidelines in the documentation before submitting pull requests.
