#!/bin/bash

# Start the API server
uvicorn app:app --host 0.0.0.0 --port $PORT
