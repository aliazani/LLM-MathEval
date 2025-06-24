# LLM-MathEval

## Overview

This project implements a comprehensive framework for fine-tuning and evaluating Large Language Models (LLMs) on mathematical reasoning tasks using the MathChat benchmark. The project includes infrastructure for fine-tuning, performance analysis, load testing, and environmental impact assessment of LLM deployments.

**Author**: Mohammadali Azani  
**Institution**: Politecnico di Milano  
**Contact**: mohammadali.azani@mail.polimi.it

## Table of Contents

- [Project Objectives](#project-objectives)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Objectives

1. **Fine-tune LLMs** for improved mathematical reasoning capabilities
2. **Identify server saturation points** to optimize deployment configurations
3. **Evaluate model performance** using the MathChat benchmark
4. **Analyze system metrics** including resource utilization and carbon emissions
5. **Provide containerized solutions** for reproducible deployments

## Architecture Overview

The project follows a three-phase approach:

1. **Fine-tuning Phase**: Adapting pre-trained LLMs for mathematical reasoning
2. **Saturation Analysis Phase**: Determining optimal server configurations
3. **Final Evaluation Phase**: Comprehensive performance and environmental impact assessment

## Project Structure

```
.
├── fine_tuning/                    # LLM fine-tuning infrastructure 
│   ├── app/
│   │   ├── entrypoint.sh           # Container entry script
│   │   ├── finetune.py             # Fine-tuning implementation
│   │   ├── data/                   # Mounted input data
│   │   │   └── mathchat.json
│   │   └── aliazn/                 # Your Hugging Face username
│   │       └── mathchat-mistral/   # Model output & checkpoints
│   │           ├── README.md
│   │           ├── adapter_config.json
│   │           ├── adapter_model.safetensors
│   │           ├── chat_template.jinja
│   │           ├── checkpoint-8155/
│   │           ├── checkpoint-16310/
│   │           ├── checkpoint-24465/
│   │           ├── special_tokens_map.json
│   │           ├── tokenizer.json
│   │           ├── tokenizer_config.json
│   │           ├── tokenizer.model
│   │           └── training_args.bin
│   ├── Dockerfile                  # Container configuration
│   ├── docker-compose.yml          # Service orchestration
│   └── .env                        # Environment variables
│
├── find_saturation_point/          # Server capacity analysis
│   ├── mathchat/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── .env                        # Environment variables
│   │   └── app/
│   │       ├── entrypoint.sh
│   │       ├── monitor.py           # Resource monitoring
│   │       └── emissions_logs/      # Created at runtime
│   │           └── system_metrics.csv
│   │
│   ├── jmeter/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── app/
│   │       ├── generate-plan.jmx    # JMeter test plan
│   │       ├── extract_questions_and_answers.py
│   │       ├── entrypoint.sh
│   │       ├── MathChat/             # Runtime output
│   │       ├── qa.jsonl              # Runtime output
│   │       ├── questions.csv         # Runtime output
│   │       ├── jmeter_run.log        # Runtime output
│   │       ├── report/               # JMeter report (Runtime output)
│   │       └── results_run.jtl       # Runtime output
│   │
│   └── load_analysis/
│       ├── requirements.txt
│       ├── max_users/
│       │   ├── analysis.py          # User capacity analysis
│       │   ├── results_elapsed_time.png # Runtime outputs
│       │   ├── results_error_rate.png   # Runtime outputs
│       │   └── results_stats.csv        # Runtime outputs
│       │
│       ├── system_metrics/
│       │   ├── resource_plot.py                # Resource visualization
│       │   ├── cpu_line_chart.png              # Runtime outputs
│       │   ├── gpu_memory_line_chart.png       # Runtime outputs
│       │   ├── gpu_utilization_line_chart.png  # Runtime outputs
│       │   └── memory_line_chart.png           # Runtime outputs
│       │
│       └── system_metrics_users/
│           ├── resource_correlator.py             # Correlate metrics with users
│           ├── correlation_combined_overview.png  # Runtime outputs
│           ├── correlation_correlated_data.csv    # Runtime outputs
│           ├── correlation_cpu_vs_load.png        # Runtime outputs
│           ├── correlation_gpu_memory_vs_load.png # Runtime outputs
│           ├── correlation_gpu_util_vs_load.png   # Runtime outputs
│           └── correlation_memory_vs_load.png     # Runtime outputs
│
└── final_run/                      # Production evaluation 
    ├── phoenix/                    # Observability platform
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── app/
    │       ├── phoenix_startup.py
    │       └── .phoenix/           # Created at runtime
    │           ├── exports/
    │           ├── inferences/
    │           ├── phoenix.db
    │           ├── phoenix.db-shm
    │           ├── phoenix.db-wal
    │           └── trace_datasets/
    │
    ├── mathchat/                   # LLM server
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   ├── .env                        # Environment variables
    │   └── app/
    │       ├── entrypoint.sh
    │       ├── monitor.py
    │       ├── server.py           # VLLM server implementation
    │       ├── __pycache__/        # Created at runtime
    │       └── emissions_logs/     # Created at runtime
    │           ├── emissions.csv
    │           └── system_metrics.csv
    │
    ├── jmeter/                     # Load testing
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── app/
    │       ├── generate-plan.jmx
    │       ├── extract_questions_and_answers.py
    │       ├── entrypoint.sh
    │       ├── MathChat/           # Created at runtime
    │       ├── qa.jsonl
    │       ├── questions.csv
    │       ├── jmeter_run1.log
    │       ├── report1/
    │       ├── results_run1.jtl
    │       ├── jmeter_run2.log
    │       ├── report2/
    │       ├── results_run2.jtl
    │       ├── jmeter_run3.log
    │       ├── report3/
    │       └── results_run3.jtl
    │
    ├── labelling/
    │   ├── evaluation/             # Model evaluation
    │   │   ├── Dockerfile
    │   │   ├── docker-compose.yml
    │   │   └── app/
    │   │       ├── answer_evaluator.py
    │   │       ├── extract_questions_and_answers.py
    │   │       ├── download_data.py
    │   │       ├── evaluate.sh
    │   │       └── output/         # Created at runtime
    │   │           ├── results.csv
    │   │           └── summary.json
    │   │
    │   └── ollama/                # Local LLM deployment
    │       ├── Dockerfile
    │       ├── docker-compose.yml
    │       ├── ollama_data/    # Created at runtime
    │       └── app/
    │           └── ollama-start.sh
    │
    └── load_analysis/             # Final-run load analyses
        ├── requirements.txt
        │
        ├── max_users/            # Created at runtime
        │   ├── analysis.py
        │   ├── run1/
        │   │   ├── results_elapsed_time.png   # Created at runtime
        │   │   ├── results_error_rate.png     # Created at runtime
        │   │   └── results_stats.csv          # Created at runtime
        │   ├── run2/
        │   │   ├── results_elapsed_time.png   # Created at runtime   
        │   │   ├── results_error_rate.png     # Created at runtime
        │   │   └── results_stats.csv          # Created at runtime
        │   └── run3/
        │       ├── results_elapsed_time.png   # Created at runtime
        │       ├── results_error_rate.png     # Created at runtime
        │       └── results_stats.csv          # Created at runtime
        │
        ├── system_metrics/       # Created at runtime
        │   ├── resource_plot.py
        │   ├── emissions_plot.py
        │   ├── cpu_plot.png                      # Created at runtime
        │   ├── gpu_mem_plot.png                  # Created at runtime
        │   ├── gpu_util_plot.png                 # Created at runtime
        │   ├── memory_plot.png                   # Created at runtime
        │   ├── summary_plot.png                  # Created at runtime
        │   ├── cpu_energy_time_series.png        # Created at runtime
        │   ├── cpu_power_time_series.png         # Created at runtime
        │   ├── emissions_rate_time_series.png    # Created at runtime
        │   ├── emissions_time_series.png         # Created at runtime
        │   ├── energy_consumed_time_series.png   # Created at runtime
        │   ├── gpu_energy_time_series.png        # Created at runtime
        │   ├── gpu_power_time_series.png         # Created at runtime
        │   ├── ram_energy_time_series.png        # Created at runtime
        │   ├── ram_power_time_series.png         # Created at runtime
        │   └── summary_time_series.png           # Created at runtime
        │
        └── system_metrics_users/ # Created at runtime
            ├── resource_correlator.py
            ├── emissions_correlator.py
            ├── emissions_cpu_energy_combined.png       # Created at runtime
            ├── emissions_cpu_power_combined.png        # Created at runtime
            ├── emissions_emissions_combined.png        # Created at runtime
            ├── emissions_emissions_rate_combined.png   # Created at runtime
            ├── emissions_energy_consumed_combined.png  # Created at runtime
            ├── emissions_gpu_energy_combined.png       # Created at runtime
            ├── emissions_gpu_power_combined.png        # Created at runtime
            ├── emissions_ram_energy_combined.png       # Created at runtime
            ├── emissions_ram_power_combined.png        # Created at runtime
            ├── correlation_avg_elapsed_combined.png    # Created at runtime
            ├── correlation_avg_error_rate_combined.png # Created at runtime
            ├── correlation_cpu_usage_combined.png      # Created at runtime
            ├── correlation_gpu_mem_combined.png        # Created at runtime
            ├── correlation_gpu_util_combined.png       # Created at runtime
            └── correlation_memory_percent_combined.png # Created at runtime
```

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.20+
- NVIDIA GPU with CUDA support (for fine-tuning and inference)
- Hugging Face account with API token
- Python 3.8+ (for analysis scripts)
- Minimum 16GB RAM (32GB recommended)
- 50GB+ available disk space

### Required Environment Variables

Create a `.env` file in each component directory with:

The First one int the fine_tuning directory:

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token
```
The Second one in the finding_saturation_point/math_chat directory:
```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token
```

The third one in the final_run/mathchat directory:

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token

# Phoenix Configuration
PHOENIX_API_KEY=your_phoenix_api_key
PHOENIX_PROJECT=final_phase
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aliazani/LLM-MathEval.git
cd LLM-MathEval
```

### 2. Run the first phase: Fine-tuning

```bash
## Phase 1: Fine-tuning
cd LLM-MathEval/fine_tuning

# Build and run the fine-tuning service
docker-compose up --build
```

### 3. Run the second phase: Finding Saturation Point

```bash
## Phase 2: Finding Saturation Point
## Run the MathChat server
cd LLM-MathEval/finding_saturation_point/mathchat
docker-compose up --build
## After the server is up we can run the JMeter tests
cd LLM-MathEval/finding_saturation_point/jmeter
docker-compose up --build

## After the JMeter tests are done, we can analyze the results
cd LLM-MathEval/finding_saturation_point/load_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python max_users/analysis.py ../jmeter/app/results_run.jtl 
## outputs: results_elapsed_time.png
#           results_error_rate.png
#           results_stats.csv     
python system_metrics/resource_plot.py ../mathchat/app/emissions_logs/system_metrics.csv 
## outputs:  resource_plot.py              
#            cpu_line_chart.png            ```           
#            gpu_memory_line_chart.png                   
#            gpu_utilization_line_chart.png## Detailed Setup and Execution Guide
#            memory_line_chart.png         

python system_metrics_users/resource_correlator.py ../mathchat/app/emissions_logs/system_metrics.csv ../jmeter/app/results_run.jtl
## outputs:  resource_correlator.py            
#            correlation_combined_overview.png
#            correlation_correlated_data.csv   
#            correlation_cpu_vs_load.png       
#            correlation_gpu_memory_vs_load.png
#            correlation_gpu_util_vs_load.png  
#            correlation_memory_vs_load.png    

deactivate

```

### 4. the third phase: Final Production Run

```bash
# Phase 3: Final Production Run
# Start the Phoenix observability platform
cd LLM-MathEval/final_run/phoenix
docker-compose up --build
# After the Phoenix is up, we can start the MathChat server

cd LLM-MathEval/final_run/mathchat
docker-compose up --build
# After the MathChat server is up, we can run the JMeter tests

cd LLM-MathEval/final_run/jmeter
docker-compose up --build

# After the JMeter tests are done, we can label the results using the ollama model
cd LLM-MathEval/labelling/ollama
docker-compose up --build
# After the ollama model is up, we can run the evaluation
cd LLM-MathEval/labelling/evaluation
docker-compose up --build

# After the evaluation is done, we can analyze the results
cd LLM-MathEval/final_run/load_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python max_users/analysis.py ../jmeter/app/results_run1.jtl
## outputs: results_elapsed_time.png
#           results_error_rate.png
#           results_stats.csv     
mv max_users/results_elapsed_time.png max_users/run1/results_elapsed_time.png
mv max_users/results_error_rate.png max_users/run1/results_error_rate.png
mv max_users/results_stats.csv max_users/run1/results_stats.csv

python max_users/analysis.py ../jmeter/app/results_run2.jtl
## outputs: results_elapsed_time.png
#           results_error_rate.png
#           results_stats.csv     

mv max_users/results_elapsed_time.png max_users/run2/results_elapsed_time.png
mv max_users/results_error_rate.png max_users/run2/results_error_rate.png
mv max_users/results_stats.csv max_users/run2/results_stats.csv

python max_users/analysis.py ../jmeter/app/results_run3.jtl
## outputs: results_elapsed_time.png
#           results_error_rate.png
#           results_stats.csv     
mv max_users/results_elapsed_time.png max_users/run3/results_elapsed_time.png
mv max_users/results_error_rate.png max_users/run3/results_error_rate.png
mv max_users/results_stats.csv max_users/run3/results_stats.csv

python system_metrics/resource_plot.py ../mathchat/app/emissions_logs/system_metrics.csv max_users/run1/results_stats.csv max_users/run2/results_stats.csv max_users/run3/results_stats.csv
## outputs: cpu_plot.png                    
#          gpu_mem_plot.png                
#          gpu_util_plot.png               
#          memory_plot.png                 
#          summary_plot.png                

python system_metrics/emissions_plot.py ../mathchat/app/emissions_logs/emissions.csv max_users/run1/results_stats.csv max_users/run2/results_stats.csv max_users/run3/results_stats.csv
## outputs: cpu_energy_time_series.png      
#           cpu_power_time_series.png       
#           emissions_rate_time_series.png  
#           emissions_time_series.png       
#           energy_consumed_time_series.png 
#           gpu_energy_time_series.png      
#           gpu_power_time_series.png       
#           ram_energy_time_series.png      
#           ram_power_time_series.png       
#           summary_time_series.png         

python system_metrics_users/resource_correlator.py ../mathchat/app/emissions_logs/system_metrics.csv ../jmeter/app/results_run1.jtl max_users/run1/results_stats.csv max_users/run2/results_stats.csv max_users/run3/results_stats.csv
## outputs: correlation_avg_elapsed_combined.png   
#           correlation_avg_error_rate_combined.png
#           correlation_cpu_usage_combined.png     
#           correlation_gpu_mem_combined.png       
#           correlation_gpu_util_combined.png      
#           correlation_memory_percent_combined.png

python system_metrics_users/emissions_correlator.py ../mathchat/app/emissions_logs/emissions.csv ../jmeter/app/results_run1.jtl max_users/run1/results_stats.csv max_users/run2/results_stats.csv max_users/run3/results_stats.csv
## outputs: emissions_cpu_energy_combined.png      
#           emissions_cpu_power_combined.png       
#           emissions_emissions_combined.png       
#           emissions_emissions_rate_combined.png  
#           emissions_energy_consumed_combined.png 
#           emissions_gpu_energy_combined.png      
#           emissions_gpu_power_combined.png       
#           emissions_ram_energy_combined.png      
#           emissions_ram_power_combined.png       

deactivate
```

## Monitoring and Observability

The project uses Phoenix for comprehensive observability:

- Real-time performance metrics
- Request tracing with OpenTelemetry

Access Phoenix dashboard at: `http://localhost:6006`

## Troubleshooting

### Emergency Cleanup

```bash
# Stop all containers
docker-compose down -v

# Remove all project containers
docker ps -a | grep -E "mathchat|phoenix|jmeter|ollama" | awk '{print $1}' | xargs -r docker rm -f

# Clean volumes
docker volume prune -f

# Remove images (optional)
docker images | grep -E "mathchat|phoenix|jmeter|ollama" | awk '{print $3}' | xargs -r docker rmi -f
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Politecnico di Milano for computational resources
- Hugging Face for model hosting
- The open-source community for invaluable tools

---

For questions or support, please contact: mohammadali.azani@mail.polimi.it