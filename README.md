# LLMPerf

A Tool for evaulation the performance of LLM APIs.

# Installation
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

# Usage Example for vLLM Testing

Install vLLM with `pip` or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

Start the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL_NAME --port $PORT --chat-template $TEMPLATE_FILE --disable-log-requests --uvicorn-log-level error
```

Run the LLMPerf test:

```bash
export OPENAI_API_KEY=EMPTY; export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1/"
python token_benchmark_ray.py --model $MODEL_NAME --mean-input-tokens $isl   --stddev-input-tokens 0 --mean-output-tokens $osl   --stddev-output-tokens 0 --max-num-completed-requests $total_requests --timeout 600 --num-concurrent-requests $concurrent_requests   --results-dir $results --llm-api openai --additional-sampling-params '{}' --model-dir $MODEL_PATH
```


### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
| `$osl` | Benchmark output sequence length. |
| `$total_requests` | Total number of requests to make. Suggestion: 5 * concurrent_requests, when concurrent_requests = 1. |
| `$concurrent_requests` | Number of concurrent requests to make. |
| `$results` | Path to store end results to. |
| `$MODEL_NAME` | Name of the model. Consistent with the model name used in the server. |
| `$MODEL_PATH` | Path to the model. |

### Model Configuration

The model should be configured to disable the generation of EOS tokens. This can be done by modifying the model configuration file. 

In `config.json`, set eos_token_id to `-1`:

```json
    "eos_token_id": -1
```

In `generation_config.json`, set eos_token_id to `[-1, -1]`:
```json
    "eos_token_id": [-1, -1],
```

In `special_tokens_map.json`, set eos_token to `<|noend_of_text|>`:
```json
    "eos_token": "<|noend_of_text|>"
```

In `tokenizer_config.json`, set eos_token to `<|noeot_id|>`:

```json
    "eos_token": "<|noeot_id|>",
```
### Example
<!-- 1,4,8,16,32,64,128,256 -->
| `$concurrent_requests` | `$total_requests` |    ISL/OSL   |
| ---------------------- | ----------------- | ------------ |
| 1                      | 5                 | 128/128      |
| 1                      | 5                 | 2048/128     |
| 1                      | 5                 | 128/2048     |
| 1                      | 5                 | 2048/2048    |
| 4                      | 4                 | 128/2048     |
| 8                      | 8                 | 128/2048     |
| 16                     | 16                | 128/2048     |
| 32                     | 32                | 128/2048     |
| 64                     | 64                | 128/2048     |
| 128                    | 128               | 128/2048     |
| 256                    | 256               | 128/2048     |

# Basic Usage

We implement 2 tests for evaluating LLMs: a load test to check for performance and a correctness test to check for correctness.

## Load test

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets. Tokens are counted using the `LlamaTokenizer` regardless of which LLM API is being tested. This is to ensure that the prompts are consistent across different LLM APIs.

To run the most basic load test you can the token_benchmark_ray script.


### Caveats and Disclaimers

- The endpoints provider backend might vary widely, so this is not a reflection on how the software runs on a particular hardware.
- The results may vary with time of day.
- The results may vary with the load.
- The results may not correlate with users’ workloads.

### OpenAI Compatible APIs
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'

```

### Anthropic
```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python token_benchmark_ray.py \
--model "claude-2" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api anthropic \
--additional-sampling-params '{}'

```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python token_benchmark_ray.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python token_benchmark_ray.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "litellm" \
--additional-sampling-params '{}'

```

### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python token_benchmark_ray.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "vertexai" \
--additional-sampling-params '{}'

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python token_benchmark_ray.py --help` for more details on the arguments.

## Correctness Test

The correctness test spawns a number of concurrent requests to the LLM API with the following format:

```
Convert the following sequence of words into a number: {random_number_in_word_format}. Output just your final answer.
```

where random_number_in_word_format could be for example "one hundred and twenty three". The test then checks that the response contains that number in digit format which in this case would be 123.

The test does this for a number of randomly generated numbers and reports the number of responses that contain a mismatch.

To run the most basic correctness test you can run the the llm_correctness.py script.

### OpenAI Compatible APIs

```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://console.endpoints.anyscale.com/m/v1

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-concurrent-requests 10 \
--results-dir "result_outputs"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=secret_abcdefg

python llm_correctness.py \
--model "claude-2" \
--llm-api "anthropic"  \
--max-num-completed-requests 5 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs"
```

### TogetherAI

```bash
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"

python llm_correctness.py \
--model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### Hugging Face

```bash
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"

python llm_correctness.py \
--model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### LiteLLM

LLMPerf can use LiteLLM to send prompts to LLM APIs. To see the environment variables to set for the provider and arguments that one should set for model and additional-sampling-params.

see the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

```bash
python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "litellm" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

see `python llm_correctness.py --help` for more details on the arguments.


### Vertex AI

Here, --model is used for logging, not for selecting the model. The model is specified in the Vertex AI Endpoint ID.

The GCLOUD_ACCESS_TOKEN needs to be somewhat regularly set, as the token generated by `gcloud auth print-access-token` expires after 15 minutes or so.

Vertex AI doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.


```bash

gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
export GCLOUD_REGION=YOUR_REGION
export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

python llm_correctness.py \
--model "meta-llama/Llama-2-7b-chat-hf" \
--llm-api "vertexai" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

### SageMaker

SageMaker doesn't return the total number of tokens that are generated by their endpoint, so tokens are counted using the LLama tokenizer.

```bash

export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"s
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"

python llm_correctness.py \
--model "llama-2-7b" \
--llm-api "sagemaker" \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \

```

## Saving Results

The results of the load test and correctness test are saved in the results directory specified by the `--results-dir` argument. The results are saved in 2 files, one with the summary metrics of the test, and one with metrics from each individual request that is returned.

# Advanced Usage

The correctness tests were implemented with the following workflow in mind:

```python
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher


# Copying the environment variables and passing them to ray.init() is necessary
# For making any clients work.
ray.init(runtime_env={"env_vars": {"OPENAI_API_BASE" : "https://api.endpoints.anyscale.com/v1",
                                   "OPENAI_API_KEY" : "YOUR_API_KEY"}})

base_prompt = "hello_world"
tokenizer = LlamaTokenizerFast.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# Create a client for spawning requests
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompt
    )

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)

```

# Implementing New LLM Clients

To implement a new LLM client, you need to implement the base class `llmperf.ray_llm_client.LLMClient` and decorate it as a ray actor.

```python

from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class CustomLLMClient(LLMClient):

    def llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...

```

# Legacy Codebase
The old LLMPerf code base can be found in the [llmperf-legacy](https://github.com/ray-project/llmval-legacy) repo.
