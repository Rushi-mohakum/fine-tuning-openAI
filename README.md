# fine-tuning-openAI
# OpenAI Fine-tuning in Standalone Jupyter Notebook

## Installation and Setup

### Option 1: Using Anaconda (Recommended)
```bash
# Install Anaconda from https://www.anaconda.com/products/distribution
# Then create environment
conda create -n openai_env python=3.9
conda activate openai_env
conda install jupyter notebook
pip install openai pandas numpy scikit-learn
```

### Option 2: Using pip
```bash
pip install jupyter notebook
pip install openai pandas numpy scikit-learn
```

### Option 3: Using Google Colab (No Installation Required)
- Go to https://colab.research.google.com
- Create new notebook
- All packages are pre-installed except OpenAI

---

## Starting Jupyter Notebook

### Local Installation:
```bash
# Navigate to your project folder
cd your_project_folder

# Start Jupyter
jupyter notebook
# or
jupyter lab
```

### Google Colab:
- Just open https://colab.research.google.com
- Create new notebook

---

## Cell 1: Environment Setup and Installation

```python
# Cell 1: Check environment and install packages
import sys
import subprocess
import os

print(f"Python version: {sys.version}")
print(f"Running on: {sys.platform}")

# For Google Colab
if 'google.colab' in sys.modules:
    print("🔧 Running on Google Colab")
    # Install packages
    !pip install openai --quiet
    # Mount Google Drive (optional)
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print("🔧 Running on local Jupyter")

# Install packages if needed
required_packages = ['openai', 'pandas', 'numpy', 'scikit-learn']

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} is available")
    except ImportError:
        print(f"Installing {package}...")
        !pip install {package}

print("✅ Environment setup complete!")
```

---

## Cell 2: Import Libraries

```python
# Cell 2: Import all required libraries
import openai
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # For visualizations
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')

print("✅ All libraries imported successfully!")
print(f"OpenAI version: {openai.__version__}")
```

---

## Cell 3: API Key Setup (Jupyter-Specific)

```python
# Cell 3: API Key Setup
import getpass
from IPython.display import display, HTML

# Method 1: Secure input (recommended)
print("🔐 Enter your OpenAI API key securely:")
api_key = getpass.getpass("API Key: ")

# Method 2: For Google Colab with secrets
if 'google.colab' in sys.modules:
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        print("✅ API key loaded from Colab secrets")
    except:
        print("⚠️ API key not found in Colab secrets")
        api_key = getpass.getpass("Enter API Key: ")

# Method 3: Environment variable
if not api_key:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ API key loaded from environment")

# Initialize client
if api_key:
    client = openai.OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")
    
    # Test connection
    try:
        models = client.models.list()
        print(f"✅ Connection successful! {len(models.data)} models available")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
else:
    print("❌ No API key provided")
```

---

## Cell 4: Enhanced Fine-tuning Class with Jupyter Features

```python
# Cell 4: Enhanced Fine-tuning class for Jupyter
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from tqdm.notebook import tqdm

class JupyterOpenAIFineTuner:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.training_stats = {}
    
    def prepare_data_with_analysis(self, data: List[Dict], output_file: str = "training_data.jsonl"):
        """Prepare data with analysis for Jupyter"""
        # Data analysis
        prompts = [item["prompt"] for item in data]
        completions = [item["completion"] for item in data]
        
        # Statistics
        prompt_lengths = [len(p.split()) for p in prompts]
        completion_lengths = [len(c.split()) for c in completions]
        
        stats = {
            "total_examples": len(data),
            "avg_prompt_length": np.mean(prompt_lengths),
            "avg_completion_length": np.mean(completion_lengths),
            "total_tokens_estimate": sum(prompt_lengths) + sum(completion_lengths)
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(prompt_lengths, bins=20, alpha=0.7, color='blue')
        axes[0].set_title('Prompt Length Distribution')
        axes[0].set_xlabel('Words')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(completion_lengths, bins=20, alpha=0.7, color='green')
        axes[1].set_title('Completion Length Distribution')
        axes[1].set_xlabel('Words')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Display stats
        print("📊 Dataset Analysis:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Save data
        with open(output_file, 'w') as f:
            for item in data:
                training_example = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["completion"]}
                    ]
                }
                f.write(json.dumps(training_example) + '\n')
        
        print(f"✅ Training data saved to {output_file}")
        self.training_stats = stats
        return output_file
    
    def upload_with_progress(self, file_path: str):
        """Upload file with progress indicator"""
        try:
            print("📤 Uploading training file...")
            with open(file_path, 'rb') as f:
                response = self.client.files.create(file=f, purpose='fine-tune')
            
            print(f"✅ File uploaded successfully!")
            print(f"   File ID: {response.id}")
            print(f"   Filename: {response.filename}")
            print(f"   Size: {response.bytes} bytes")
            return response.id
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None
    
    def create_job_with_options(self, training_file_id: str, model: str = "gpt-3.5-turbo", 
                               custom_hyperparams: Dict = None):
        """Create fine-tuning job with customizable options"""
        
        # Default hyperparameters
        hyperparams = {
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 2
        }
        
        # Update with custom parameters
        if custom_hyperparams:
            hyperparams.update(custom_hyperparams)
        
        print("🚀 Creating fine-tuning job with parameters:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model,
                hyperparameters=hyperparams
            )
            
            print(f"✅ Fine-tuning job created!")
            print(f"   Job ID: {response.id}")
            print(f"   Model: {response.model}")
            print(f"   Status: {response.status}")
            
            return response.id
        except Exception as e:
            print(f"❌ Job creation failed: {e}")
            return None
    
    def monitor_with_progress(self, job_id: str, update_interval: int = 30):
        """Monitor job with live progress updates"""
        print("🔄 Monitoring fine-tuning progress...")
        print("   (This cell will update automatically)")
        
        start_time = time.time()
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                
                # Clear output and show current status
                clear_output(wait=True)
                
                elapsed = time.time() - start_time
                print(f"🔄 Fine-tuning Progress")
                print(f"   Job ID: {job_id}")
                print(f"   Status: {job.status}")
                print(f"   Model: {job.model}")
                print(f"   Elapsed time: {elapsed/60:.1f} minutes")
                
                if hasattr(job, 'trained_tokens') and job.trained_tokens:
                    print(f"   Trained tokens: {job.trained_tokens:,}")
                
                if job.status == "succeeded":
                    print(f"\n🎉 Fine-tuning completed successfully!")
                    print(f"   Fine-tuned model: {job.fine_tuned_model}")
                    print(f"   Total time: {elapsed/60:.1f} minutes")
                    return job.fine_tuned_model
                
                elif job.status == "failed":
                    print(f"\n❌ Fine-tuning failed!")
                    if hasattr(job, 'error') and job.error:
                        print(f"   Error: {job.error}")
                    return None
                
                elif job.status in ["cancelled", "canceled"]:
                    print(f"\n⚠️ Fine-tuning was cancelled")
                    return None
                
                # Wait before next check
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\n⚠️ Monitoring interrupted by user")
                return None
            except Exception as e:
                print(f"\n❌ Monitoring error: {e}")
                time.sleep(update_interval)
    
    def test_model_interactive(self, model_id: str, test_prompts: List[str]):
        """Test model with interactive results display"""
        results = []
        
        print(f"🧪 Testing fine-tuned model: {model_id}")
        print("=" * 50)
        
        for i, prompt in enumerate(tqdm(test_prompts, desc="Testing")):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                result = {
                    "prompt": prompt,
                    "completion": response.choices[0].message.content,
                    "tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
                results.append(result)
                
                # Display result
                print(f"\n📝 Test {i+1}:")
                print(f"   Prompt: {prompt}")
                print(f"   Response: {result['completion']}")
                print(f"   Tokens: {result['tokens']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
                
            except Exception as e:
                print(f"❌ Error testing prompt {i+1}: {e}")
        
        # Summary statistics
        if results:
            total_tokens = sum(r['tokens'] for r in results)
            avg_tokens = total_tokens / len(results)
            
            print(f"\n📊 Testing Summary:")
            print(f"   Total tests: {len(test_prompts)}")
            print(f"   Successful: {len(results)}")
            print(f"   Total tokens used: {total_tokens:,}")
            print(f"   Average tokens per test: {avg_tokens:.1f}")
        
        return results

print("✅ Enhanced Jupyter fine-tuning class ready!")
```

---

## Cell 5: Create and Analyze Sample Data

```python
# Cell 5: Create comprehensive sample dataset
def create_comprehensive_dataset():
    """Create a more comprehensive sample dataset"""
    base_data = [
        {"prompt": "What is machine learning?", "completion": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."},
        {"prompt": "Explain neural networks", "completion": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions."},
        {"prompt": "What is deep learning?", "completion": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data."},
        {"prompt": "Define supervised learning", "completion": "Supervised learning is a machine learning approach where algorithms learn from labeled training data to make predictions or decisions on new, unseen data."},
        {"prompt": "What is unsupervised learning?", "completion": "Unsupervised learning is a machine learning technique that finds hidden patterns in data without using labeled examples or target outputs."},
    ]
    
    # Expand with variations
    expanded_data = []
    variations = [
        "Can you explain {}?",
        "Tell me about {}",
        "What do you know about {}?",
        "Give me information on {}",
        "Help me understand {}"
    ]
    
    for item in base_data:
        expanded_data.append(item)
        # Add 2-3 variations for each base prompt
        base_topic = item["prompt"].lower().replace("what is ", "").replace("define ", "").replace("explain ", "")
        for variation in variations[:2]:  # Use first 2 variations
            new_prompt = variation.format(base_topic)
            expanded_data.append({
                "prompt": new_prompt,
                "completion": item["completion"]
            })
    
    return expanded_data

# Create dataset
sample_data = create_comprehensive_dataset()
print(f"✅ Created dataset with {len(sample_data)} examples")

# Initialize fine-tuner
fine_tuner = JupyterOpenAIFineTuner(api_key)

# Prepare data with analysis
training_file = fine_tuner.prepare_data_with_analysis(sample_data, "jupyter_training_data.jsonl")
```

---

## Cell 6: Upload and Start Fine-tuning

```python
# Cell 6: Upload file and create job
print("🚀 Starting fine-tuning process...")

# Upload training file
file_id = fine_tuner.upload_with_progress("jupyter_training_data.jsonl")

if file_id:
    # Custom hyperparameters (optional)
    custom_params = {
        "n_epochs": 3,
        "batch_size": 1,
        "learning_rate_multiplier": 2
    }
    
    # Create fine-tuning job
    job_id = fine_tuner.create_job_with_options(
        file_id, 
        model="gpt-3.5-turbo",
        custom_hyperparams=custom_params
    )
    
    if job_id:
        print(f"\n✅ Ready to monitor job: {job_id}")
        print("Run the next cell to start monitoring...")
    else:
        print("❌ Failed to create fine-tuning job")
else:
    print("❌ Failed to upload training file")
```

---

## Cell 7: Monitor Progress (Interactive)

```python
# Cell 7: Monitor fine-tuning with live updates
# Update this with your actual job ID from the previous cell
job_id = "ftjob-your-job-id-here"  # Replace with actual job ID

if job_id != "ftjob-your-job-id-here":
    model_id = fine_tuner.monitor_with_progress(job_id, update_interval=30)
    
    if model_id:
        print(f"\n🎉 Fine-tuning completed!")
        print(f"Your model ID: {model_id}")
        
        # Save model ID
        with open("fine_tuned_model_id.txt", "w") as f:
            f.write(model_id)
        print("✅ Model ID saved to 'fine_tuned_model_id.txt'")
    else:
        print("❌ Fine-tuning did not complete successfully")
else:
    print("⚠️ Please update the job_id variable with your actual job ID")
```

---

## Cell 8: Test the Fine-tuned Model

```python
# Cell 8: Test your fine-tuned model
# Load or enter your model ID
try:
    with open("fine_tuned_model_id.txt", "r") as f:
        model_id = f.read().strip()
    print(f"✅ Loaded model ID: {model_id}")
except:
    model_id = input("Enter your fine-tuned model ID: ").strip()

if model_id:
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain reinforcement learning",
        "What are the benefits of machine learning?",
        "Tell me about computer vision",
        "What is natural language processing?"
    ]
    
    # Test the model
    test_results = fine_tuner.test_model_interactive(model_id, test_prompts)
    
    # Create results DataFrame for analysis
    if test_results:
        df_results = pd.DataFrame(test_results)
        print("\n📊 Results DataFrame:")
        display(df_results)
        
        # Visualize token usage
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(test_results)), [r['tokens'] for r in test_results])
        plt.title('Token Usage per Test')
        plt.xlabel('Test Number')
        plt.ylabel('Total Tokens')
        
        plt.subplot(1, 2, 2)
        prompt_tokens = [r['prompt_tokens'] for r in test_results]
        completion_tokens = [r['completion_tokens'] for r in test_results]
        
        x = range(len(test_results))
        plt.bar(x, prompt_tokens, label='Prompt', alpha=0.7)
        plt.bar(x, completion_tokens, bottom=prompt_tokens, label='Completion', alpha=0.7)
        plt.title('Token Breakdown')
        plt.xlabel('Test Number')
        plt.ylabel('Tokens')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
else:
    print("❌ No model ID provided")
```

---

## Cell 9: Cost Analysis and Comparison

```python
# Cell 9: Comprehensive cost analysis
def calculate_detailed_costs(training_stats, test_results=None):
    """Calculate detailed cost analysis"""
    
    # Training costs (approximate)
    training_cost_per_1k = 0.0080  # GPT-3.5-turbo
    training_tokens = training_stats.get('total_tokens_estimate', 0)
    training_cost = (training_tokens / 1000) * training_cost_per_1k
    
    # Usage costs
    usage_cost_per_1k = 0.0015  # Fine-tuned model usage
    
    costs = {
        'training': {
            'tokens': training_tokens,
            'cost': training_cost,
            'rate_per_1k': training_cost_per_1k
        }
    }
    
    if test_results:
        total_usage_tokens = sum(r['tokens'] for r in test_results)
        usage_cost = (total_usage_tokens / 1000) * usage_cost_per_1k
        
        costs['usage'] = {
            'tokens': total_usage_tokens,
            'cost': usage_cost,
            'rate_per_1k': usage_cost_per_1k
        }
        
        costs['total'] = training_cost + usage_cost
    else:
        costs['total'] = training_cost
    
    return costs

# Calculate costs
if 'fine_tuner' in locals() and hasattr(fine_tuner, 'training_stats'):
    cost_analysis = calculate_detailed_costs(
        fine_tuner.training_stats, 
        test_results if 'test_results' in locals() else None
    )
    
    print("💰 Cost Analysis")
    print("=" * 30)
    print(f"Training:")
    print(f"  Tokens: {cost_analysis['training']['tokens']:,}")
    print(f"  Cost: ${cost_analysis['training']['cost']:.4f}")
    
    if 'usage' in cost_analysis:
        print(f"\nUsage (Testing):")
        print(f"  Tokens: {cost_analysis['usage']['tokens']:,}")
        print(f"  Cost: ${cost_analysis['usage']['cost']:.4f}")
    
    print(f"\nTotal Estimated Cost: ${cost_analysis['total']:.4f}")
    
    # Visualization
    if 'usage' in cost_analysis:
        labels = ['Training', 'Usage']
        costs = [cost_analysis['training']['cost'], cost_analysis['usage']['cost']]
        
        plt.figure(figsize=(8, 6))
        plt.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Cost Breakdown')
        plt.axis('equal')
        plt.show()
```

---

## Cell 10: Save Results and Next Steps

```python
# Cell 10: Save all results and provide next steps
import pickle
from datetime import datetime

# Create results summary
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'training_file': 'jupyter_training_data.jsonl',
    'model_id': model_id if 'model_id' in locals() else None,
    'job_id': job_id if 'job_id' in locals() else None,
    'training_stats': fine_tuner.training_stats if hasattr(fine_tuner, 'training_stats') else {},
    'test_results': test_results if 'test_results' in locals() else [],
    'cost_analysis': cost_analysis if 'cost_analysis' in locals() else {}
}

# Save to files
with open('fine_tuning_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save DataFrame if it exists
if 'df_results' in locals():
    df_results.to_csv('test_results.csv', index=False)

print("💾 Results saved to:")
print("  - fine_tuning_results.json")
if 'df_results' in locals():
    print("  - test_results.csv")

print("\n🚀 Next Steps:")
print("1. Experiment with different hyperparameters")
print("2. Create a larger, more diverse dataset")
print("3. Set up evaluation metrics")
print("4. Compare with base model performance")
print("5. Deploy for production use")

print("\n📚 Useful Resources:")
print("- OpenAI Fine-tuning Docs: https://platform.openai.com/docs/guides/fine-tuning")
print("- API Reference: https://platform.openai.com/docs/api-reference/fine-tuning")
print("- Community Forum: https://community.openai.com/")

# Display final summary
if results_summary['model_id']:
    print(f"\n🎉 Your Fine-tuned Model ID: {results_summary['model_id']}")
    print("You can now use this model in your applications!")
```

---

## Key Differences from VS Code:

### 1. **Installation**
- Standalone Jupyter requires manual package installation
- Google Colab has most packages pre-installed

### 2. **API Key Management**
- Use `getpass` for secure input
- Google Colab has built-in secrets management
- Environment variables work the same way

### 3. **Enhanced Features**
- **Progress bars** with `tqdm.notebook`
- **Interactive widgets** with `ipywidgets`
- **Rich visualizations** with matplotlib/seaborn
- **Live output updates** with `clear_output()`

### 4. **File Management**
- Files are saved in the notebook's working directory
- Google Colab can mount Google Drive for persistence

### 5. **Monitoring**
- Real-time progress updates in cells
- Visual progress indicators
- Automatic output clearing and updating

## Tips for Jupyter Notebook:

1. **Use `Shift+Enter`** to run cells
2. **Use `Ctrl+Enter`** to run current cell without moving
3. **Use `Alt+Enter`** to run cell and insert new cell below
4. **Save frequently** with `Ctrl+S`
5. **Use markdown cells** for documentation
6. **Restart kernel** if you encounter issues
