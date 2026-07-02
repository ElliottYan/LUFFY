<div align="center">


<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
  <img src="./figures/logo.png" alt="LUFFY Icon" width="50">
  LUFFY: Learning to Reason Under Off‑Policy Guidance
</h1>
<p align="center"><em>A general framework for off-policy learning in large reasoning models.</em></p>

<div align="center">
  <img src="./figures/luffy_intro_new.jpg" alt="overview" style="width: 66%; height: auto;">
</div>


[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2504.14945) [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)](https://www.alphaxiv.org/abs/2504.14945) [![Github](https://img.shields.io/badge/LUFFY-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ElliottYan/LUFFY)   [![Hugging Face Collection](https://img.shields.io/badge/LUFFY_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/yafuly/status/1914559433549676962)





</div>

---

# 📚 Overview
- 🎉 [News](#news)  
- 📖 [Introduction](#introduction)  
- ✨ [Getting Started](#getting-started)  
- 🔧 [Usage](#usage)  
- 📃 [Evaluation](#evaluation)  
- 🎈 [Citation](#citation)  
- 🌻 [Acknowledgement](#acknowledgement)  
<!-- - 📈 [Star History](#star-history) -->


---


# 🎉News
- **[2026/01/26]** 🎉 ExGRPO has been accepted to **ICLR 2026**!
- **[2025/10/04]** 🚀 Introducing [ExGRPO](https://github.com/ElliottYan/LUFFY/tree/main/ExGRPO), a new variant that boosts performance by **learning from the model’s own off-policy experience**, without relying on external guidance.
- **[2025/09/19]** 🎉 LUFFY has been accepted to **NeurIPS 2025**!
- **[2025/05/30]** We integrate the implementation and scripts of other off-policy learning methods including SFT, SFT+RL and RL w/ SFT Loss (multi-task learning).
- **[2025/05/21]** We have updated the paper [version](https://arxiv.org/abs/2504.14945), which re-evaluates all models using a more accurate verifier and adds comparisons with other off-policy learning methods, including RL with SFT Loss and SFT+RL.
- **[2025/04/23]** Our paper now trending on [alphaXiv](https://www.alphaxiv.org/abs/2504.14945)! We welcome feedback and discussion.
- **[2025/04/23]** 🎉 Ranked **#1** of the day on [Huggingface Daily Papers](https://huggingface.co/papers/2504.14945).
- **[2025/04/20]** LUFFY paper available on [arXiv](http://arxiv.org/abs/2504.14945). 

<!-- - **[2025/04/20]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4).
- **[2025/04/20]** LUFFY codebase is released along with evaluation scripts. Try it out! -->

---

# 📖Introduction

LUFFY is a reinforcement learning framework that bridges the gap between zero-RL and imitation learning by incorporating off-policy reasoning traces into the training process. Built upon GRPO, LUFFY combines on-policy rollouts with off-policy demonstrations during advantage estimation and introduces **policy shaping** via regularized importance sampling to emphasize low-probability yet crucial actions.

![overview](./figures/luffy_performance.jpg)

### Key Highlights:
- **Off-Policy Guidance:** Seamlessly integrates external reasoning traces to bootstrap learning from stronger models.
- **Dynamic Balance:** Learns when to imitate and when to explore, adapting over the course of training.
- **Policy Shaping:** Emphasizes important actions often ignored in standard policy gradients, enabling better generalization.



---

# ✨Getting Started

## Installation

You can install LUFFY dependencies by running the following commands:
```bash
conda create -n luffy python=3.10
conda activate luffy
cd luffy
pip install -r requirements.txt
pip install -e .
cd verl
pip install -e .
```

### Update 9.8
Recently, we find the deprecation of pyairports caused a lot of environment issues. Thus, we now provide a bit more complicated way for installing the environment. 
```bash
conda create -n luffy python=3.10
conda activate luffy
pip install airports-py
git clone https://github.com/dottxt-ai/outlines.git
cd outlines
git checkout 0.0.46
pip install .
cd ../luffy
pip install -r requirements.v2.txt
pip install -e .
cd verl
pip install -e .
```


If you encounter issues when installing flash-attn, we recommend you to install it here 
[flash-attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3). For example, we use this version. 
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Repo Structure

This repository includes:

- `luffy`: Codes for training LUFFY using off-policy reasoning traces. Our main code changes are in luffy/verl/verl/mix_src.
- `data`: Data and code for training and evaluating LUFFY. 
- `exp_scripts`: Example script to train LUFFY.
- `eval_scripts`: Evaluation scripts on math and out-of-distribution benchmarks.
- `ExGRPO`: Implementation and notes for ExGRPO, which leverages off-policy experience replay to further boost performance without external guidance.

LUFFY is built on top of the GRPO framework and supports plug-and-play integration with off-policy traces from models such as DeepSeek-R1.

---





# 🔧Usage

## Data Preparation
You need to first run the data preparation script to get the training data in parquet format.
```bash
cd data
python prepare_train.py
```

## Training

We provide an example script to train LUFFY on our subset of OpenR1-Math-220k. You can run the following command to train LUFFY:

```bash
  cd exp_scripts
  bash train.sh
```

## Other Off-Policy Baselines
### SFT
First clone the OpenRLHF repository and prepare the data to SFT format. *(We plan to integrate the SFT pipeline directly into LUFFY in the near future.)*
```bash
git clone https://github.com/OpenRLHF/OpenRLHF
cd data
python prepare_sft.py
```
Then, you can run the SFT training command. 
```
RESULT_DIR="Your result directory"
DATA_DIR="Your data directory"
WANDB_KEY="Your Wandb Key"

MODEL_PATH=Elliott/Qwen2.5-Math-7B-16k-think
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
DEVICES="0,1,2,3,4,5,6,7"
deepspeed --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --include localhost:$DEVICES --module openrlhf.cli.train_sft \
   --max_len 16384 \
   --dataset $DATA_DIR \
   --input_key prompt \
   --output_key target \
   --train_batch_size 64 \
   --apply_chat_template \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain $MODEL_PATH \
   --save_path $RESULT_DIR \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --adam_offload \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --save_hf_ckpt \
   --learning_rate 5e-5 \
   --lr_warmup_ratio 0.1 \
   --wandb_project r1_sft_distill \
   --wandb_run_name qwen-7b-base-sft \
   --use_wandb $WANDB_KEY \
   --gradient_checkpointing
```


### RL w/ SFT Loss
```bash
  cd exp_scripts
  bash train_rl_sft_loss.sh
```

### SFT + RL
We use heldout data for RL training, following previous works like PRIME.
```bash
  cd data
  python prepare_train_sft_rl.py
  cd ../exp_scripts
  bash train_sft_rl.sh
```

## Inference

Here’s an example of using LUFFY for inference:

<details>
<summary>Click to view inference example</summary>

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path="Elliott/LUFFY-Qwen-Math-7B-Zero"

question = "which number is larger? 9.11 or 9.9?"

tokenizer = AutoTokenizer.from_pretrained(model_path)
messages = [{"role": "user", "content": question}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_path)
params = SamplingParams(temperature=0.6, max_tokens=8192)
outputs = llm.generate([chat], params)
print(outputs[0].outputs[0].text)
```

</details>


## Models

| **Model**                          | **Huggingface** |  **Base Model** |
|-----------------------------------|------------------|------------------|
| LUFFY-Qwen-Math-7B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-7B-Zero |  Qwen2.5-Math-7B |
| LUFFY-Qwen-Math-7B-SFT | https://huggingface.co/Elliott/Qwen2.5-Math-7B-SFT | Qwen2.5-Math-7B |
| LUFFY-Qwen-Math-7B-SFT-RL | https://huggingface.co/Elliott/Qwen2.5-Math-7B-SFT-RL | Qwen2.5-Math-7B |
| LUFFY-Qwen-Math-1.5B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-1.5B-Zero | Qwen2.5-Math-1.5B |
| LUFFY-Qwen-Instruct-7B | https://huggingface.co/Elliott/LUFFY-Qwen-Instruct-7B | Qwen2.5-7B-Instruct |

---

# 📃Evaluation

## Reproducing the Results 
We currently support automated evaluation on six widely used mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). The platform provides specialized system prompts for a range of RL models, including LUFFY, SimpleRL, OpenReasoner, PRIME, and OAT.

You can reproduce our results by running the following commands:
```bash
ROOT=YOUR_ROOT_PATH
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=Elliott/LUFFY-Qwen-Math-7B-Zero
MODEL_NAME=luffy

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
```



## LUFFY on Qwen2.5-Math-7B (zero-RL)
LUFFY is evaluated on six competition-level benchmarks, achieving state-of-the-art results among all zero-RL methods. It surpasses both on-policy RL and imitation learning (SFT), especially in generalization:



| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-Math-7B                      |11.5 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 |
| Qwen2.5-Math-7B-Instruct             |12.5  | 10.2 | 48.5 | 80.4 | 32.7 | 41.0 | 37.6   |
| SimpleRL-Zero                     | 27.0 | 6.8  | 54.9 | 76.0 | 25.0 | 34.7 | 37.4     |
| OpenReasoner-Zero                 | 16.5 | 15.0 | 52.1 | 82.4 | 33.1 | 47.1 | 41.0    |
| PRIME-Zero                        | 17.0 | 12.8 | 54.0 | 81.4 | **39.0** | 40.3 | 40.7    |
| Oat-Zero                          | **33.4**  | 11.9 | 61.2 | 78.0 | 34.6 | 43.4 | 43.7   |
| **LUFFY-Qwen-Math-7B-Zero**                         | 29.4        | **23.1**        | **65.6**| **87.6**      | 37.5        | **57.2**      | **50.1** |

---



LUFFY also generalizes well to out-of-distribution tasks, with over +6.2 average gain on ARC-C, GPQA, and MMLU-Pro.


| **Model**                         | **ARC-c** | **GPQA-diamond** | **MMLU-Pro** | **Avg.** |
|----------------------------------|-----------|------------------|--------------|----------|
| Qwen2.5-Math-7B             | 18.2 | 11.1 | 16.9 | 15.4  |
| Qwen2.5-Math-7B-Instruct         | 70.3 | 24.7 | 34.1 | 43.0    |
| SimpleRL-Zero                    | 30.2 | 23.2 | 34.5 | 29.3     |
| OpenReasoner-Zero                       | 66.2 | 29.8 | 58.7 | 51.6     |
| PRIME-Zero                         | 73.3 | 18.2 | 32.7 | 41.4   |
| Oat-Zero                | 70.1 | 23.7 | 41.7 | 45.2    |
| **LUFFY-Qwen-Math-7B-Zero**                        | **80.5** |  **39.9** | **53.0** | **57.8** |


We further compare LUFFY with alternative off-policy learning methods, including SFT, RL w/ SFT Loss and SFT+RL (see our paper for details):

| **Model**                          | **GPU Hours** | **Data Usage (On/Off)** | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| SFT                      | 24*8 | 0 / 64k | 22.2 | 22.3 | 52.8 | 82.6 | 40.8 | 43.7 | 44.1 |
| RL w/ SFT Loss             |  133*8    | 64k*7 / 64k  |  19.5 | 16.4 | 49.7 | 80.4 | 34.9 | 39.4 | 40.1  |
| SFT+RL                      | 130*8 |  64k*8/135k |  25.8 | **23.1** | 62.7 | 87.2 | 39.7 | 50.4 | 48.2 |
| **LUFFY-Qwen-Math-7B-Zero**            | 77*8 | 64k*7 / 64k              | 29.4        | **23.1**        | 65.6 | **87.6**      | 37.5        | **57.2**      | 50.1 |
| **LUFFY-Qwen-Math-7B-Zero-Extra**       |    130*8       |   110k*7 / 110k      | **30.7** | 22.5 | **66.2**|  86.8 | **41.2** | 55.3 | **50.4** |

---

## LUFFY on Qwen2.5-Math-1.5B
| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-Math-1.5B                  |   7.2 |  3.6 | 26.4 | 28.0 | 9.6 | 21.2 | 16.0 |
| Qwen2.5-Math-1.5B-Instruct            |  12.1 | 8.9 | 48.1 | 77.4 | 28.7 | 39.1 | 35.7 |
| **LUFFY-Qwen-Math-1.5B-Zero**             | **16.0** | **13.1** | **47.1** | **80.2** | **30.5** | **41.0** | **38.0** |



## LUFFY on Qwen2.5-Instruct-7B 
| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-7B-Instruct           | 11.7 | 7.5 | 43.8 | 71.8 | 30.9 | 40.4|  34.4|
| **LUFFY-Qwen-Instruct-7B**             | **17.7** |  **14.8** | **50.9** | **82.0** | **31.3** | **47.4** | **40.7** |


# 🌻Acknowledgement

LUFFY builds upon [veRL](https://github.com/volcengine/verl) and [deepscaler](https://github.com/agentica-project/rllm), and utilizes [vLLM](https://github.com/vllm-project/vllm) for inference. We utilize [Math-Verify](https://github.com/huggingface/Math-Verify) for math reasoning evaluation. We thank the open-source community for datasets and backbones, including [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [DeepSeek-R1](https://github.com/deepseek-ai/deepseek-r1) model. 

# 📬 Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:
- Jianhao Yan: elliottyan37@gmail.com
- Yafu Li: yafuly@gmail.com

# Citation
If you find our model, data, or evaluation code useful, please kindly cite our paper.


**LUFFY**:
```bib
@misc{luffy,
      title={Learning to Reason under Off-Policy Guidance}, 
      author={Jianhao Yan and Yafu Li and Zican Hu and Zhi Wang and Ganqu Cui and Xiaoye Qu and Yu Cheng and Yue Zhang},
      year={2025},
      eprint={2504.14945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14945}, 
}
```


**ExGRPO**:
```bib
@article{zhan2025exgrpo,
      title={ExGRPO: Learning to Reason from Experience}, 
      author={Runzhe Zhan and Yafu Li and Zhi Wang and Xiaoye Qu and Dongrui Liu and Jing Shao and Derek F. Wong and Yu Cheng},
      year={2025},
      journal = {ArXiv preprint},
      volume = {2510.02245},
      url={https://arxiv.org/abs/2510.02245}, 
}
```


### 📝 Complete TODO List

#### root/
- `test.py:1590`: add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged

#### verl/examples/split_placement/
- `verl/examples/split_placement/split_monkey_patch.py:141`: make a canonical logger that supports various backend

#### verl/tests/e2e/
- `verl/tests/e2e/check_results.py:21`: this function needs error handling

#### verl/tests/ray/
- `verl/tests/ray/test_high_level_scheduling_api.py:25`: pass *args and **kwargs is bug prone and not very convincing
- `verl/tests/ray/test_worker_group_basics.py:43`: pass *args and **kwargs is bug prone and not very convincing

#### verl/verl/
- `verl/verl/protocol.py:260`: we can actually lift this restriction if needed

#### verl/verl/mix_src/
- `verl/verl/mix_src/mix_fsdp_worker.py:83`: it seems that manual offload is slowly than FSDP offload
- `verl/verl/mix_src/mix_fsdp_worker.py:207`: add transformer policy We force reference policy to use CPUOffload to save memory. We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
- `verl/verl/mix_src/mix_fsdp_worker.py:226`: add more optimizer args into config
- `verl/verl/mix_src/mix_fsdp_worker.py:263`: a sharding manager that do nothing?
- `verl/verl/mix_src/mix_fsdp_worker.py:391`: here, we should return all metrics
- `verl/verl/mix_src/mix_fsdp_worker.py:517`: support DCP and save sharded checkpoints
- `verl/verl/mix_src/mix_trainer.py:90`: add other ways to estimate advantages
- `verl/verl/mix_src/mix_trainer.py:168`: support each role have individual ray_worker_group_cls, i.e., support different backend of different role
- `verl/verl/mix_src/mix_trainer.py:293`: we have to make sure the batch size is divisible by the dp size
- `verl/verl/mix_src/mix_trainer.py:599`: make a canonical logger that supports various backend
- `verl/verl/mix_src/mix_trainer.py:637`: add response length
- `verl/verl/mix_src/mix_trainer_acc_rebatch.py:63`: we have to make sure the batch size is divisible by the dp size
- `verl/verl/mix_src/mix_trainer_acc_rebatch.py:437`: make a canonical logger that supports various backend
- `verl/verl/mix_src/mix_trainer_acc_rebatch.py:592`: check path
- `verl/verl/mix_src/mix_trainer_acc_rebatch.py:628`: from remote not implemented yet

#### verl/verl/models/llama/megatron/
- `verl/verl/models/llama/megatron/modeling_llama_megatron.py:330`: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
- `verl/verl/models/llama/megatron/modeling_llama_megatron.py:588`: for better performance, the sp padding should be removed at each layer. Not sure the performance gap

#### verl/verl/models/llama/megatron/layers/
- `verl/verl/models/llama/megatron/layers/parallel_attention.py:380`: llama does not have dropout in the config?? It is recommended to use dropout with FA according to the docs when training.
- `verl/verl/models/llama/megatron/layers/parallel_decoder.py:78`: add sequence parallel operator reduce_scatter here
- `verl/verl/models/llama/megatron/layers/parallel_decoder.py:86`: add sequence parallel operator all_gather here
- `verl/verl/models/llama/megatron/layers/parallel_decoder.py:90`: add sequence parallel operator reduce_scatter here

#### verl/verl/models/transformers/
- `verl/verl/models/transformers/llama.py:88`: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache to be able to avoid many of these transpose/reshape/view.

#### verl/verl/single_controller/ray/
- `verl/verl/single_controller/ray/base.py:439`: create a class with customizable name

#### verl/verl/third_party/vllm/vllm_v_0_3_1/
- `verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:101`: currently is hfconfig
- `verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:145`: check get_lora_tokenizer func
- `verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:586`: check this input
- `verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:661`: we may not need to decode
- `verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py:62`: check megatron
- `verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py:84`: need to implement a general way to deal with prefix
- `verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py:109`: do not use cupy

#### verl/verl/third_party/vllm/vllm_v_0_4_2/
- `verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py:257`: spec config
- `verl/verl/third_party/vllm/vllm_v_0_4_2/config.py:136`: for multimodal model
- `verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:130`: currently is hfconfig
- `verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:145`: check tokenizer class
- `verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:153`: don't know what's the usage
- `verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:237`: check whether we should rebuild the CUDAGraph every iter when offload/load KVCache Re-capture CUDAGraph would be time-consuming
- `verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py:67`: check megatron
- `verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py:254`: need to implement a general way to deal with prefix
- `verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py:337`: remove dependencies from megatron
- `verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py:236`: this will hang cpu_group = torch.distributed.new_group(, backend="gloo") if rank == 0: print(f'rank: {rank}') print(f'ranks: {ranks}') print(f'torch.distributed.get_process_group_ranks(shard_group): {torch.distributed.get_process_group_ranks(shard_group)}') if rank in ranks:
- `verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py:245`: will hang when used with device mesh
- `verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py:247`: init using device mesh Build the pipeline model-parallel groups.

#### verl/verl/third_party/vllm/vllm_v_0_5_4/
- `verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py:366`: spec config
- `verl/verl/third_party/vllm/vllm_v_0_5_4/config.py:191`: check whether this is necessary
- `verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py:148`: check usagecontext
- `verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py:271`: check whether we should rebuild the CUDAGraph every iter when offload/load KVCache Re-capture CUDAGraph would be time-consuming
- `verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py:67`: check megatron
- `verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py:254`: need to implement a general way to deal with prefix
- `verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:138`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:165`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:177`: init using device mesh (not support hybrid engine now) Build the pipeline model-parallel groups.
- `verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:249`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:253`: init using device mesh (not support hybrid engine now) Build the pipeline model-parallel groups.
- `verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py:84`: we don't need driver if parallel_config and is_driver_worker: assert rank % parallel_config.tensor_parallel_size == 0, \ "Driver worker should be rank 0 of tensor parallel group."

#### verl/verl/third_party/vllm/vllm_v_0_6_3/
- `verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py:147`: check usagecontext
- `verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py:345`: check whether we should rebuild the CUDAGraph every iter when offload/load KVCache Re-capture CUDAGraph would be time-consuming
- `verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py:68`: check megatron
- `verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py:255`: need to implement a general way to deal with prefix
- `verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:144`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:172`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:185`: init using device mesh (not support hybrid engine now) Build the pipeline model-parallel groups.
- `verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:257`: check why True is not work in Ray trainer
- `verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:262`: init using device mesh (not support hybrid engine now) Build the pipeline model-parallel groups.
- `verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py:92`: we don't need driver if parallel_config and is_driver_worker: assert rank % parallel_config.tensor_parallel_size == 0, \ "Driver worker should be rank 0 of tensor parallel group."

#### verl/verl/trainer/
- `verl/verl/trainer/fsdp_sft_trainer.py:77`: add checkpoint manager
- `verl/verl/trainer/fsdp_sft_trainer.py:316`: add a unified tracking

#### verl/verl/trainer/ppo/
- `verl/verl/trainer/ppo/ray_trainer.py:129`: add other ways to estimate advantages
- `verl/verl/trainer/ppo/ray_trainer.py:207`: add response length
- `verl/verl/trainer/ppo/ray_trainer.py:330`: support each role have individual ray_worker_group_cls, i.e., support different backend of different role
- `verl/verl/trainer/ppo/ray_trainer.py:379`: we have to make sure the batch size is divisible by the dp size
- `verl/verl/trainer/ppo/ray_trainer.py:632`: check path
- `verl/verl/trainer/ppo/ray_trainer.py:667`: from remote not implemented yet
- `verl/verl/trainer/ppo/ray_trainer.py:880`: make a canonical logger that supports various backend

#### verl/verl/utils/
- `verl/verl/utils/model.py:164`: we can make this faster
- `verl/verl/utils/model.py:272`: to find a better way to load mistral7b-rm lm_head
- `verl/verl/utils/torch_functional.py:362`: add them back if top_k is not None and top_k > 0: logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits) if top_p is not None and top_p < 1.0 and top_p > 0.0: logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)

#### verl/verl/utils/checkpoint/
- `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py:101`: shall we remove previous ckpt every save?
- `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py:135`: address optimizer is None

#### verl/verl/workers/
- `verl/verl/workers/fsdp_workers.py:117`: it seems that manual offload is slowly than FSDP offload
- `verl/verl/workers/fsdp_workers.py:233`: add transformer policy We force reference policy to use CPUOffload to save memory. We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
- `verl/verl/workers/fsdp_workers.py:252`: add more optimizer args into config
- `verl/verl/workers/fsdp_workers.py:289`: a sharding manager that do nothing?
- `verl/verl/workers/fsdp_workers.py:416`: here, we should return all metrics
- `verl/verl/workers/megatron_workers.py:204`: add more optimizer args into config
- `verl/verl/workers/megatron_workers.py:338`: here, we should return all metrics
- `verl/verl/workers/megatron_workers.py:478`: support vpp here vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model this_megatron_config = copy.deepcopy(megatron_config) this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
- `verl/verl/workers/megatron_workers.py:507`: add more optimizer args into config
- `verl/verl/workers/megatron_workers.py:667`: add more optimizer args into config
- `verl/verl/workers/megatron_workers.py:720`: reward model use itself tokenizer instead of sft tokenizer the input_ids, responses, attention_mask and position_ids may be different!

#### verl/verl/workers/actor/
- `verl/verl/workers/actor/megatron_actor.py:225`: actually, we just need to control the sampling order.
- `verl/verl/workers/actor/megatron_actor.py:301`: we may use the new schedule instead for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)

#### verl/verl/workers/critic/
- `verl/verl/workers/critic/megatron_critic.py:176`: we may use the new schedule instead for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)

#### verl/verl/workers/reward_model/megatron/
- `verl/verl/workers/reward_model/megatron/reward_model.py:192`: actually, we just need to control the sampling order.
- `verl/verl/workers/reward_model/megatron/reward_model.py:233`: we may use the new schedule instead for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)

#### verl/verl/workers/rollout/
- `verl/verl/workers/rollout/hf_rollout.py:98`: filter out the seq with no answers like ds-chat

#### verl/verl/workers/sharding_manager/
- `verl/verl/workers/sharding_manager/fsdp_ulysses.py:49`: check how to set seed for each model
- `verl/verl/workers/sharding_manager/fsdp_ulysses.py:56`: check how to set seed for each model
- `verl/verl/workers/sharding_manager/fsdp_vllm.py:82`: offload FSDP model weights self.module.cpu() torch.cuda.empty_cache() if torch.distributed.get_rank() == 0: print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')
- `verl/verl/workers/sharding_manager/fsdp_vllm.py:113`: Current impl doesn't consider FSDP with torch micro-dp
- `verl/verl/workers/sharding_manager/fsdp_vllm.py:122`: Current impl doesn't consider FSDP with torch micro-dp
- `verl/verl/workers/sharding_manager/fsdp_vllm.py:130`: shall we build a micro_dp group for vllm when integrating with vLLM?
- `verl/verl/workers/sharding_manager/megatron_vllm.py:76`: after binding to the memory buffer, we can load the checkpoint here

