#!/bin/bash
set -e

# Print colored messages
print_green() { printf "\e[32m%s\e[0m\n" "$1"; }
print_yellow() { printf "\e[33m%s\e[0m\n" "$1"; }
print_red() { printf "\e[31m%s\e[0m\n" "$1"; }

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_yellow "Creating .env file. Please edit it with your API keys after script completes."
    cat > .env << EOL
GOOGLE_API_KEY= a9khSPPH7f_hsgUm8_u-07WLkWlKIPHQf5VCQiS1E-0WcuJ0wOL62A
AGENTQL_API_KEY= AIzaSyCQj7zGtdaeGXu8wWDA6jYiukkvwWFDPHI
EOL
fi

# Create necessary directories
print_green "Creating necessary directories..."
mkdir -p FinalDataSource
mkdir -p chroma_db
mkdir -p evaluation
mkdir -p data_source

# Check if processed data exists
if [ ! -f "FinalDataSource/processed_assessments.json" ]; then
    print_yellow "Processed assessment data not found."
    
    # Check if source data exists
    if [ ! -f "data_source/shl_enhanced_solutions.json" ] || [ ! -f "data_source/shl_enhanced_solutions_prepacksol.json" ]; then
        print_yellow "Source data files not found."
        
        # Prompt for data scraping
        read -p "Do you want to run the data scraping scripts now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_green "Running data scraping scripts..."
            python DataScraper.py
            python Url_DataScraper.py
            print_green "Data scraping complete."
        else
            print_yellow "Skipping data scraping. Please provide the data files manually."
        fi
    fi
    
    # Run data processing
    print_green "Processing assessment data..."
    python DataProcessor.py
    print_green "Data processing complete."
fi

# Copy test data if it doesn't exist
if [ ! -f "evaluation/test_data.json" ]; then
    print_green "Copying test data for evaluation..."
    cp test_data.json evaluation/
fi

# Deployment method choice
print_yellow "How would you like to deploy the application?"
echo "1) Run locally with Python"
echo "2) Deploy with Docker Compose"
read -p "Enter your choice (1 or 2): " deploy_choice

case $deploy_choice in
    1)
        # Local deployment
        print_green "Starting local deployment..."
        
        # Check if dependencies are installed
        if ! pip list | grep -q "fastapi"; then
            print_yellow "Installing dependencies..."
            ./setup.sh
        fi
        
        # Start server in background
        print_green "Starting API server..."
        python api_server.py &
        API_PID=$!
        
        # Wait for API to become available
        print_yellow "Waiting for API server to start..."
        sleep 5
        
        # Start Streamlit in background
        print_green "Starting Streamlit web interface..."
        streamlit run streamlit_app.py &
        STREAMLIT_PID=$!
        
        print_green "Application started!"
        print_green "API: http://localhost:8000"
        print_green "Web interface: http://localhost:8501"
        print_green "To stop the application, press Ctrl+C"
        
        # Wait for interrupt
        trap "kill $API_PID $STREAMLIT_PID; exit" INT TERM
        wait
        ;;
    2)
        # Docker deployment
        print_green "Starting Docker Compose deployment..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            print_red "Docker not found. Please install Docker and Docker Compose."
            exit 1
        fi
        
        # Build and start containers
        print_green "Building and starting containers..."
        docker-compose up --build -d
        
        print_green "Application deployed successfully!"
        print_green "API: http://localhost:8000"
        print_green "Web interface: http://localhost:8501"
        print_green "To stop the application, run: docker-compose down"
        ;;
    *)
        print_red "Invalid choice. Exiting."
        exit 1
        ;;
esac