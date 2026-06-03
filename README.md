# LUFFY: Learning to Reason Under Off‑Policy Guidance

.. (first part: title, badges, overview, news, introduction, getting started, usage, evaluation tables) ...

# 🌻Acknowledgement

### 📝 Complete TODO List

1. **ExGRPO/exgrpo/verl/tests/ray/test_high_level_scheduling_api.py** (line 25)
   - pass *args and **kwargs is bug prone and not very convincing
2. **ExGRPO/exgrpo/verl/tests/ray/test_worker_group_basics.py** (line 43)
   - pass *args and **kwargs is bug prone and not very convincing
3. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 54)
   - support FSDP hybrid shard for larger model
4. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 83)
   - it seems that manual offload is slowly than FSDP offload
5. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 123)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
6. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 199)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
7. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 207)
   - add transformer policy
8. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 226)
   - add more optimizer args into config
9. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 252)
   - support FSDP hybrid shard for larger model
10. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 263)
   - a sharding manager that do nothing?
11. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 426)
   - here, we should return all metrics
12. **ExGRPO/exgrpo/verl/verl/mix_src/mix_fsdp_worker.py** (line 586)
   - support DCP and save sharded checkpoints
13. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer.py** (line 90)
   - add other ways to estimate advantages
14. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer.py** (line 150)
   - support each role have individual ray_worker_group_cls,
15. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer.py** (line 197)
   - we have to make sure the batch size is divisible by the dp size
16. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer.py** (line 508)
   - make a canonical logger that supports various backend
17. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer.py** (line 552)
   - add response length
18. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 63)
   - we have to make sure the batch size is divisible by the dp size
19. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 437)
   - make a canonical logger that supports various backend
20. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 592)
   - check path
21. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 628)
   - from remote not implemented yet
22. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_experience.py** (line 64)
   - support each role have individual ray_worker_group_cls,
23. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_experience.py** (line 534)
   - make a canonical logger that supports various backend
24. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_helper.py** (line 40)
   - add other ways to estimate advantages
25. **ExGRPO/exgrpo/verl/verl/mix_src/mix_trainer_helper.py** (line 97)
   - add response length
26. **ExGRPO/exgrpo/verl/verl/mix_src/mix_vllm_rollout.py** (line 43)
   - implementation needed
27. **ExGRPO/exgrpo/verl/verl/mix_src/mix_vllm_rollout_exp.py** (line 43)
   - implementation needed
28. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/layers/parallel_attention.py** (line 380)
   - llama does not have dropout in the config??
29. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 78)
   - add sequence parallel operator reduce_scatter here
30. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 86)
   - add sequence parallel operator all_gather here
31. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 90)
   - add sequence parallel operator reduce_scatter here
32. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/modeling_llama_megatron.py** (line 330)
   - for better performance, the sp padding should be removed at each layer. Not sure the performance gap
33. **ExGRPO/exgrpo/verl/verl/models/llama/megatron/modeling_llama_megatron.py** (line 588)
   - for better performance, the sp padding should be removed at each layer. Not sure the performance gap
34. **ExGRPO/exgrpo/verl/verl/models/registry.py** (line 21)
   - HF may supported more than listed here, we should add more after testing
35. **ExGRPO/exgrpo/verl/verl/models/transformers/llama.py** (line 88)
   - These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
36. **ExGRPO/exgrpo/verl/verl/protocol.py** (line 164)
   - (zhangchi.usc1992) add consistency check
37. **ExGRPO/exgrpo/verl/verl/protocol.py** (line 260)
   - we can actually lift this restriction if needed
38. **ExGRPO/exgrpo/verl/verl/protocol.py** (line 346)
   - (zhangchi.usc1992) whether to copy
39. **ExGRPO/exgrpo/verl/verl/single_controller/ray/base.py** (line 439)
   - create a class with customizable name
40. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py** (line 64)
   - delete the unused args
41. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py** (line 147)
   - Support fine-grained seeds (e.g., seed per request).
42. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 237)
   - maybe we can hack the autoregressive logics without only apply post process for better performance
43. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 241)
   - we can optimize it by making the dataloader yield List[int] without padding.
44. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 257)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
45. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 99)
   - Print more configs in debug mode.
46. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 101)
   - currently is hfconfig
47. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 112)
   - maybe we can choose init here or from arguments
48. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 145)
   - check get_lora_tokenizer func
49. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 586)
   - check this input
50. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 661)
   - we may not need to decode
51. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 67)
   - latest commit in vllm fix awq for this function and add load_weights
52. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 96)
   - (pad to be divided by 4)
53. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 224)
   - Change the get_logits part to a separate stage.
54. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/tokenizer.py** (line 56)
   - the lora tokenizer is also passed, but may be different
55. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py** (line 62)
   - check megatron
56. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py** (line 84)
   - need to implement a general way to deal with prefix
57. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 109)
   - do not use cupy
58. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 209)
   - Profile swapping overhead and optimize if needed.
59. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 291)
   - maybe we should also flag the megatron is initialized
60. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 44)
   - implementation needed
61. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 109)
   - delete the unused args
62. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 192)
   - Support fine-grained seeds (e.g., seed per request).
63. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 257)
   - spec config
64. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/config.py** (line 136)
   - for multimodal model
65. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/hf_weight_loader.py** (line 81)
   - implementation needed
66. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 268)
   - maybe we can hack the autoregressive logics without only apply post process for better performance
67. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 272)
   - we can optimize it by making the dataloader yield List[int] without padding.
68. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 288)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
69. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 128)
   - Print more configs in debug mode.
70. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 130)
   - currently is hfconfig
71. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 143)
   - maybe we can choose init here or from arguments
72. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 145)
   - check tokenizer class
73. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 153)
   - don't know what's the usage
74. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 228)
   - add for verl but we may not tokenizer in Rollout
75. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 237)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
76. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 67)
   - check megatron
77. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 254)
   - need to implement a general way to deal with prefix
78. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 272)
   - latest commit in vllm fix awq for this function and add load_weights
79. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 325)
   - (pad to be divided by 4)
80. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 337)
   - remove dependencies from megatron
81. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py** (line 141)
   - This is a hack, we need to register the load_weight() func for each model in vllm
82. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py** (line 226)
   - This is a hack, we need to register the load_weight() func for each model in vllm
83. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/model_runner.py** (line 274)
   - perform sampling on rank 0
84. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 236)
   - this will hang
85. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 245)
   - will hang when used with device mesh
86. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 247)
   - init using device mesh
87. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py** (line 62)
   - verl not support speculative decode now
88. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py** (line 208)
   - not implemented async executor yet
89. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/tokenizer.py** (line 61)
   - the lora tokenizer is also passed, but may be different
90. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py** (line 30)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
91. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py** (line 270)
   - check whether need this
92. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 53)
   - check this
93. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 54)
   - check this
94. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 143)
   - delete the unused args
95. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 226)
   - Support fine-grained seeds (e.g., seed per request).
96. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 366)
   - spec config
97. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/config.py** (line 191)
   - check whether this is necessary
98. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/hf_weight_loader.py** (line 32)
   - implementation needed
99. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 148)
   - check usagecontext
100. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 205)
   - we can optimize it by making the dataloader yield List[int] without padding.
101. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 221)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
102. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 143)
   - Print more configs in debug mode.
103. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 160)
   - maybe we can choose init here or from arguments
104. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 262)
   - add for verl but we may not tokenizer in Rollout
105. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 271)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
106. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 67)
   - check megatron
107. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 254)
   - need to implement a general way to deal with prefix
108. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 272)
   - latest commit in vllm fix awq for this function and add load_weights
109. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py** (line 152)
   - This is a hack, we need to register the load_weight() func for each model in vllm
110. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py** (line 239)
   - This is a hack, we need to register the load_weight() func for each model in vllm
111. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 94)
   - deviate from the v0.5.4, not pp now
112. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 138)
   - check why True is not work in Ray trainer
113. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 165)
   - check why True is not work in Ray trainer
114. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 177)
   - init using device mesh (not support hybrid engine now)
115. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 249)
   - check why True is not work in Ray trainer
116. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 253)
   - init using device mesh (not support hybrid engine now)
117. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py** (line 65)
   - verl not support speculative decode now
118. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py** (line 243)
   - not implemented async executor yet
119. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/tokenizer.py** (line 61)
   - the lora tokenizer is also passed, but may be different
120. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 29)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
121. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 84)
   - we don't need driver
122. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 103)
   - set correct model runner class
123. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 301)
   - check whether need this
124. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/hf_weight_loader.py** (line 29)
   - implementation needed
125. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 147)
   - check usagecontext
126. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 170)
   - we can optimize it by making the dataloader yield List[int] without padding.
127. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 186)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
128. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 174)
   - Print more configs in debug mode.
129. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 336)
   - add for verl but we may not tokenizer in Rollout
130. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 345)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
131. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 68)
   - check megatron
132. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 255)
   - need to implement a general way to deal with prefix
133. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 273)
   - latest commit in vllm fix awq for this function and add load_weights
134. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py** (line 170)
   - This is a hack, we need to register the load_weight() func for each model in vllm
135. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py** (line 273)
   - This is a hack, we need to register the load_weight() func for each model in vllm
136. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 97)
   - deviate from the v0.5.4, not pp now
137. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 144)
   - check why True is not work in Ray trainer
138. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 172)
   - check why True is not work in Ray trainer
139. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 185)
   - init using device mesh (not support hybrid engine now)
140. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 257)
   - check why True is not work in Ray trainer
141. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 262)
   - init using device mesh (not support hybrid engine now)
142. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py** (line 73)
   - verl not support speculative decode now
143. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py** (line 246)
   - not implemented async executor yet
144. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 33)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
145. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 92)
   - we don't need driver
146. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 110)
   - set correct model runner class
147. **ExGRPO/exgrpo/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 311)
   - check whether need this
148. **ExGRPO/exgrpo/verl/verl/trainer/fsdp_sft_trainer.py** (line 77)
   - add checkpoint manager
149. **ExGRPO/exgrpo/verl/verl/trainer/fsdp_sft_trainer.py** (line 140)
   - implementation needed
150. **ExGRPO/exgrpo/verl/verl/trainer/fsdp_sft_trainer.py** (line 316)
   - add a unified tracking
151. **ExGRPO/exgrpo/verl/verl/trainer/fsdp_sft_trainer.py** (line 333)
   - (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
152. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 129)
   - add other ways to estimate advantages
153. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 207)
   - add response length
154. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 330)
   - support each role have individual ray_worker_group_cls,
155. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 379)
   - we have to make sure the batch size is divisible by the dp size
156. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 632)
   - check path
157. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 667)
   - from remote not implemented yet
158. **ExGRPO/exgrpo/verl/verl/trainer/ppo/ray_trainer.py** (line 885)
   - make a canonical logger that supports various backend
159. **ExGRPO/exgrpo/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py** (line 101)
   - shall we remove previous ckpt every save?
160. **ExGRPO/exgrpo/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py** (line 135)
   - address optimizer is None
161. **ExGRPO/exgrpo/verl/verl/utils/hdfs_io.py** (line 67)
   - implementation needed
162. **ExGRPO/exgrpo/verl/verl/utils/hdfs_io.py** (line 102)
   - implementation needed
163. **ExGRPO/exgrpo/verl/verl/utils/megatron_utils.py** (line 202)
   - check how to disable megatron timers
164. **ExGRPO/exgrpo/verl/verl/utils/model.py** (line 164)
   - we can make this faster
165. **ExGRPO/exgrpo/verl/verl/utils/model.py** (line 272)
   - to find a better way to load mistral7b-rm lm_head
166. **ExGRPO/exgrpo/verl/verl/utils/torch_functional.py** (line 375)
   - add them back
167. **ExGRPO/exgrpo/verl/verl/workers/actor/megatron_actor.py** (line 158)
   - actually, this function should only return log_prob and this logic should be handled by user outside
168. **ExGRPO/exgrpo/verl/verl/workers/actor/megatron_actor.py** (line 225)
   - actually, we just need to control the sampling order.
169. **ExGRPO/exgrpo/verl/verl/workers/actor/megatron_actor.py** (line 301)
   - we may use the new schedule instead
170. **ExGRPO/exgrpo/verl/verl/workers/critic/megatron_critic.py** (line 176)
   - we may use the new schedule instead
171. **ExGRPO/exgrpo/verl/verl/workers/critic/megatron_critic.py** (line 176)
   - we may use the new schedule instead
172. **ExGRPO/exgrpo/verl/verl/workers/critic/megatron_critic.py** (line 176)
   - we may use the new schedule instead
173. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 88)
   - support FSDP hybrid shard for larger model
174. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 88)
   - support FSDP hybrid shard for larger model
175. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 88)
   - support FSDP hybrid shard for larger model
176. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 117)
   - it seems that manual offload is slowly than FSDP offload
177. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 117)
   - it seems that manual offload is slowly than FSDP offload
178. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 117)
   - it seems that manual offload is slowly than FSDP offload
179. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 157)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
180. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 157)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
181. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 157)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
182. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 225)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
183. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 225)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
184. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 225)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
185. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 233)
   - add transformer policy
186. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 233)
   - add transformer policy
187. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 233)
   - add transformer policy
188. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 252)
   - add more optimizer args into config
189. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 252)
   - add more optimizer args into config
190. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 252)
   - add more optimizer args into config
191. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 278)
   - support FSDP hybrid shard for larger model
192. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 278)
   - support FSDP hybrid shard for larger model
193. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 278)
   - support FSDP hybrid shard for larger model
194. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 289)
   - a sharding manager that do nothing?
195. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 289)
   - a sharding manager that do nothing?
196. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 289)
   - a sharding manager that do nothing?
197. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 416)
   - here, we should return all metrics
198. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 416)
   - here, we should return all metrics
199. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 416)
   - here, we should return all metrics
200. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 811)
   - we may need to extract it to dp_reward_model.py
201. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 811)
   - we may need to extract it to dp_reward_model.py
202. **ExGRPO/exgrpo/verl/verl/workers/fsdp_workers.py** (line 811)
   - we may need to extract it to dp_reward_model.py
203. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 106)
   - Currently, we only support reference model param offload
204. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 106)
   - Currently, we only support reference model param offload
205. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 106)
   - Currently, we only support reference model param offload
206. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 204)
   - add more optimizer args into config
207. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 204)
   - add more optimizer args into config
208. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 204)
   - add more optimizer args into config
209. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 338)
   - here, we should return all metrics
210. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 338)
   - here, we should return all metrics
211. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 338)
   - here, we should return all metrics
212. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 444)
   - support critic model offload
213. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 444)
   - support critic model offload
214. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 444)
   - support critic model offload
215. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 478)
   - support vpp here
216. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 478)
   - support vpp here
217. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 478)
   - support vpp here
218. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 507)
   - add more optimizer args into config
219. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 507)
   - add more optimizer args into config
220. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 507)
   - add more optimizer args into config
221. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 667)
   - add more optimizer args into config
222. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 667)
   - add more optimizer args into config
223. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 667)
   - add more optimizer args into config
224. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 720)
   - reward model use itself tokenizer instead of sft tokenizer
225. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 720)
   - reward model use itself tokenizer instead of sft tokenizer
226. **ExGRPO/exgrpo/verl/verl/workers/megatron_workers.py** (line 720)
   - reward model use itself tokenizer instead of sft tokenizer
227. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 145)
   - check why is bfloat16
228. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 145)
   - check why is bfloat16
229. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 145)
   - check why is bfloat16
230. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 192)
   - actually, we just need to control the sampling order.
231. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 192)
   - actually, we just need to control the sampling order.
232. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 192)
   - actually, we just need to control the sampling order.
233. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 233)
   - we may use the new schedule instead
234. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 233)
   - we may use the new schedule instead
235. **ExGRPO/exgrpo/verl/verl/workers/reward_model/megatron/reward_model.py** (line 233)
   - we may use the new schedule instead
236. **ExGRPO/exgrpo/verl/verl/workers/rollout/hf_rollout.py** (line 98)
   - filter out the seq with no answers like ds-chat
237. **ExGRPO/exgrpo/verl/verl/workers/rollout/hf_rollout.py** (line 98)
   - filter out the seq with no answers like ds-chat
238. **ExGRPO/exgrpo/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py** (line 43)
   - implementation needed
239. **ExGRPO/exgrpo/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py** (line 43)
   - implementation needed
240. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 49)
   - check how to set seed for each model
241. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 49)
   - check how to set seed for each model
242. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 56)
   - check how to set seed for each model
243. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 56)
   - check how to set seed for each model
244. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 82)
   - offload FSDP model weights
245. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 82)
   - offload FSDP model weights
246. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 113)
   - Current impl doesn't consider FSDP with torch micro-dp
247. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 113)
   - Current impl doesn't consider FSDP with torch micro-dp
248. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 122)
   - Current impl doesn't consider FSDP with torch micro-dp
249. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 122)
   - Current impl doesn't consider FSDP with torch micro-dp
250. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 130)
   - shall we build a micro_dp group for vllm when integrating with vLLM?
251. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 130)
   - shall we build a micro_dp group for vllm when integrating with vLLM?
252. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 76)
   - after binding to the memory buffer, we can load the checkpoint here
253. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 76)
   - after binding to the memory buffer, we can load the checkpoint here
254. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 253)
   - this may not be true for FSDP -> vLLM
255. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 253)
   - this may not be true for FSDP -> vLLM
256. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 323)
   - (zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
257. **ExGRPO/exgrpo/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 323)
   - (zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
258. **luffy/test.py** (line 1590)
   - add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
259. **luffy/test.py** (line 1590)
   - add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
260. **luffy/verl/examples/split_placement/split_monkey_patch.py** (line 141)
   - make a canonical logger that supports various backend
261. **luffy/verl/tests/e2e/check_results.py** (line 21)
   - this function needs error handling
262. **luffy/verl/tests/model/test_transformer.py** (line 22)
   - add more models for test
263. **luffy/verl/tests/model/test_transformer.py** (line 50)
   - we can construct the position_ids_rmpad here
264. **luffy/verl/tests/model/test_transformer.py** (line 111)
   - we can construct the position_ids_rmpad here
265. **luffy/verl/tests/model/test_transformers_ulysses.py** (line 34)
   - add more models for test
266. **luffy/verl/tests/model/test_transformers_ulysses.py** (line 81)
   - we can construct the position_ids_rmpad here
267. **luffy/verl/tests/model/test_transformers_ulysses.py** (line 159)
   - we can construct the position_ids_rmpad here
268. **luffy/verl/tests/ray/test_high_level_scheduling_api.py** (line 25)
   - pass *args and **kwargs is bug prone and not very convincing
269. **luffy/verl/tests/ray/test_worker_group_basics.py** (line 43)
   - pass *args and **kwargs is bug prone and not very convincing
270. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 54)
   - support FSDP hybrid shard for larger model
271. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 83)
   - it seems that manual offload is slowly than FSDP offload
272. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 123)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
273. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 199)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
274. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 207)
   - add transformer policy
275. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 226)
   - add more optimizer args into config
276. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 252)
   - support FSDP hybrid shard for larger model
277. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 263)
   - a sharding manager that do nothing?
278. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 391)
   - here, we should return all metrics
279. **luffy/verl/verl/mix_src/mix_fsdp_worker.py** (line 517)
   - support DCP and save sharded checkpoints
280. **luffy/verl/verl/mix_src/mix_trainer.py** (line 90)
   - add other ways to estimate advantages
281. **luffy/verl/verl/mix_src/mix_trainer.py** (line 168)
   - support each role have individual ray_worker_group_cls,
282. **luffy/verl/verl/mix_src/mix_trainer.py** (line 293)
   - we have to make sure the batch size is divisible by the dp size
283. **luffy/verl/verl/mix_src/mix_trainer.py** (line 599)
   - make a canonical logger that supports various backend
284. **luffy/verl/verl/mix_src/mix_trainer.py** (line 637)
   - add response length
285. **luffy/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 63)
   - we have to make sure the batch size is divisible by the dp size
286. **luffy/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 437)
   - make a canonical logger that supports various backend
287. **luffy/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 592)
   - check path
288. **luffy/verl/verl/mix_src/mix_trainer_acc_rebatch.py** (line 628)
   - from remote not implemented yet
289. **luffy/verl/verl/mix_src/mix_vllm_rollout.py** (line 43)
   - implementation needed
290. **luffy/verl/verl/models/llama/megatron/layers/parallel_attention.py** (line 380)
   - llama does not have dropout in the config??
291. **luffy/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 78)
   - add sequence parallel operator reduce_scatter here
292. **luffy/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 86)
   - add sequence parallel operator all_gather here
293. **luffy/verl/verl/models/llama/megatron/layers/parallel_decoder.py** (line 90)
   - add sequence parallel operator reduce_scatter here
294. **luffy/verl/verl/models/llama/megatron/modeling_llama_megatron.py** (line 330)
   - for better performance, the sp padding should be removed at each layer. Not sure the performance gap
295. **luffy/verl/verl/models/llama/megatron/modeling_llama_megatron.py** (line 588)
   - for better performance, the sp padding should be removed at each layer. Not sure the performance gap
296. **luffy/verl/verl/models/registry.py** (line 21)
   - HF may supported more than listed here, we should add more after testing
297. **luffy/verl/verl/models/transformers/llama.py** (line 88)
   - These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
298. **luffy/verl/verl/protocol.py** (line 164)
   - (zhangchi.usc1992) add consistency check
299. **luffy/verl/verl/protocol.py** (line 260)
   - we can actually lift this restriction if needed
300. **luffy/verl/verl/protocol.py** (line 346)
   - (zhangchi.usc1992) whether to copy
301. **luffy/verl/verl/single_controller/ray/base.py** (line 439)
   - create a class with customizable name
302. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py** (line 64)
   - delete the unused args
303. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py** (line 147)
   - Support fine-grained seeds (e.g., seed per request).
304. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 237)
   - maybe we can hack the autoregressive logics without only apply post process for better performance
305. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 241)
   - we can optimize it by making the dataloader yield List[int] without padding.
306. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py** (line 257)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
307. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 99)
   - Print more configs in debug mode.
308. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 101)
   - currently is hfconfig
309. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 112)
   - maybe we can choose init here or from arguments
310. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 145)
   - check get_lora_tokenizer func
311. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 586)
   - check this input
312. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py** (line 661)
   - we may not need to decode
313. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 67)
   - latest commit in vllm fix awq for this function and add load_weights
314. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 96)
   - (pad to be divided by 4)
315. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py** (line 224)
   - Change the get_logits part to a separate stage.
316. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/tokenizer.py** (line 56)
   - the lora tokenizer is also passed, but may be different
317. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py** (line 62)
   - check megatron
318. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/weight_loaders.py** (line 84)
   - need to implement a general way to deal with prefix
319. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 109)
   - do not use cupy
320. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 209)
   - Profile swapping overhead and optimize if needed.
321. **luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py** (line 291)
   - maybe we should also flag the megatron is initialized
322. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 44)
   - implementation needed
323. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 109)
   - delete the unused args
324. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 192)
   - Support fine-grained seeds (e.g., seed per request).
325. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py** (line 257)
   - spec config
326. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/config.py** (line 136)
   - for multimodal model
327. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/hf_weight_loader.py** (line 81)
   - implementation needed
328. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 268)
   - maybe we can hack the autoregressive logics without only apply post process for better performance
329. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 272)
   - we can optimize it by making the dataloader yield List[int] without padding.
330. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py** (line 288)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
331. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 128)
   - Print more configs in debug mode.
332. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 130)
   - currently is hfconfig
333. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 143)
   - maybe we can choose init here or from arguments
334. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 145)
   - check tokenizer class
335. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 153)
   - don't know what's the usage
336. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 228)
   - add for verl but we may not tokenizer in Rollout
337. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py** (line 237)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
338. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 67)
   - check megatron
339. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 254)
   - need to implement a general way to deal with prefix
340. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 272)
   - latest commit in vllm fix awq for this function and add load_weights
341. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 325)
   - (pad to be divided by 4)
342. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/megatron_weight_loaders.py** (line 337)
   - remove dependencies from megatron
343. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py** (line 141)
   - This is a hack, we need to register the load_weight() func for each model in vllm
344. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py** (line 226)
   - This is a hack, we need to register the load_weight() func for each model in vllm
345. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_runner.py** (line 274)
   - perform sampling on rank 0
346. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 236)
   - this will hang
347. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 245)
   - will hang when used with device mesh
348. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/parallel_state.py** (line 247)
   - init using device mesh
349. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py** (line 62)
   - verl not support speculative decode now
350. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py** (line 208)
   - not implemented async executor yet
351. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/tokenizer.py** (line 61)
   - the lora tokenizer is also passed, but may be different
352. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py** (line 30)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
353. **luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py** (line 270)
   - check whether need this
354. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 53)
   - check this
355. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 54)
   - check this
356. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 143)
   - delete the unused args
357. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 226)
   - Support fine-grained seeds (e.g., seed per request).
358. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py** (line 366)
   - spec config
359. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/config.py** (line 191)
   - check whether this is necessary
360. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/hf_weight_loader.py** (line 32)
   - implementation needed
361. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 148)
   - check usagecontext
362. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 205)
   - we can optimize it by making the dataloader yield List[int] without padding.
363. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py** (line 221)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
364. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 143)
   - Print more configs in debug mode.
365. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 160)
   - maybe we can choose init here or from arguments
366. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 262)
   - add for verl but we may not tokenizer in Rollout
367. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py** (line 271)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
368. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 67)
   - check megatron
369. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 254)
   - need to implement a general way to deal with prefix
370. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/megatron_weight_loaders.py** (line 272)
   - latest commit in vllm fix awq for this function and add load_weights
371. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py** (line 152)
   - This is a hack, we need to register the load_weight() func for each model in vllm
372. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py** (line 239)
   - This is a hack, we need to register the load_weight() func for each model in vllm
373. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 94)
   - deviate from the v0.5.4, not pp now
374. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 138)
   - check why True is not work in Ray trainer
375. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 165)
   - check why True is not work in Ray trainer
376. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 177)
   - init using device mesh (not support hybrid engine now)
377. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 249)
   - check why True is not work in Ray trainer
378. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py** (line 253)
   - init using device mesh (not support hybrid engine now)
379. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py** (line 65)
   - verl not support speculative decode now
380. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py** (line 243)
   - not implemented async executor yet
381. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/tokenizer.py** (line 61)
   - the lora tokenizer is also passed, but may be different
382. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 29)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
383. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 84)
   - we don't need driver
384. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 103)
   - set correct model runner class
385. **luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py** (line 301)
   - check whether need this
386. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/hf_weight_loader.py** (line 29)
   - implementation needed
387. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 147)
   - check usagecontext
388. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 170)
   - we can optimize it by making the dataloader yield List[int] without padding.
389. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py** (line 186)
   - can be optimzied by rewrite the Sampler._get_logprobs() logits
390. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 174)
   - Print more configs in debug mode.
391. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 336)
   - add for verl but we may not tokenizer in Rollout
392. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py** (line 345)
   - check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
393. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 68)
   - check megatron
394. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 255)
   - need to implement a general way to deal with prefix
395. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/megatron_weight_loaders.py** (line 273)
   - latest commit in vllm fix awq for this function and add load_weights
396. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py** (line 170)
   - This is a hack, we need to register the load_weight() func for each model in vllm
397. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py** (line 273)
   - This is a hack, we need to register the load_weight() func for each model in vllm
398. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 97)
   - deviate from the v0.5.4, not pp now
399. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 144)
   - check why True is not work in Ray trainer
400. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 172)
   - check why True is not work in Ray trainer
401. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 185)
   - init using device mesh (not support hybrid engine now)
402. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 257)
   - check why True is not work in Ray trainer
403. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py** (line 262)
   - init using device mesh (not support hybrid engine now)
404. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py** (line 73)
   - verl not support speculative decode now
405. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py** (line 246)
   - not implemented async executor yet
406. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 33)
   - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
407. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 92)
   - we don't need driver
408. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 110)
   - set correct model runner class
409. **luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py** (line 311)
   - check whether need this
410. **luffy/verl/verl/trainer/fsdp_sft_trainer.py** (line 77)
   - add checkpoint manager
411. **luffy/verl/verl/trainer/fsdp_sft_trainer.py** (line 140)
   - implementation needed
412. **luffy/verl/verl/trainer/fsdp_sft_trainer.py** (line 316)
   - add a unified tracking
413. **luffy/verl/verl/trainer/fsdp_sft_trainer.py** (line 333)
   - (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
414. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 129)
   - add other ways to estimate advantages
415. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 207)
   - add response length
416. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 330)
   - support each role have individual ray_worker_group_cls,
417. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 379)
   - we have to make sure the batch size is divisible by the dp size
418. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 632)
   - check path
419. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 667)
   - from remote not implemented yet
420. **luffy/verl/verl/trainer/ppo/ray_trainer.py** (line 880)
   - make a canonical logger that supports various backend
421. **luffy/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py** (line 101)
   - shall we remove previous ckpt every save?
422. **luffy/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py** (line 135)
   - address optimizer is None
423. **luffy/verl/verl/utils/hdfs_io.py** (line 67)
   - implementation needed
424. **luffy/verl/verl/utils/hdfs_io.py** (line 102)
   - implementation needed
425. **luffy/verl/verl/utils/megatron_utils.py** (line 202)
   - check how to disable megatron timers
426. **luffy/verl/verl/utils/model.py** (line 164)
   - we can make this faster
427. **luffy/verl/verl/utils/model.py** (line 272)
   - to find a better way to load mistral7b-rm lm_head
428. **luffy/verl/verl/utils/torch_functional.py** (line 362)
   - add them back
429. **luffy/verl/verl/workers/actor/megatron_actor.py** (line 158)
   - actually, this function should only return log_prob and this logic should be handled by user outside
430. **luffy/verl/verl/workers/actor/megatron_actor.py** (line 225)
   - actually, we just need to control the sampling order.
431. **luffy/verl/verl/workers/actor/megatron_actor.py** (line 301)
   - we may use the new schedule instead
432. **luffy/verl/verl/workers/critic/megatron_critic.py** (line 176)
   - we may use the new schedule instead
433. **luffy/verl/verl/workers/fsdp_workers.py** (line 88)
   - support FSDP hybrid shard for larger model
434. **luffy/verl/verl/workers/fsdp_workers.py** (line 117)
   - it seems that manual offload is slowly than FSDP offload
435. **luffy/verl/verl/workers/fsdp_workers.py** (line 157)
   - 1. support create from random initialized model. 2. Support init with FSDP directly
436. **luffy/verl/verl/workers/fsdp_workers.py** (line 225)
   - (zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
437. **luffy/verl/verl/workers/fsdp_workers.py** (line 233)
   - add transformer policy
438. **luffy/verl/verl/workers/fsdp_workers.py** (line 252)
   - add more optimizer args into config
439. **luffy/verl/verl/workers/fsdp_workers.py** (line 278)
   - support FSDP hybrid shard for larger model
440. **luffy/verl/verl/workers/fsdp_workers.py** (line 289)
   - a sharding manager that do nothing?
441. **luffy/verl/verl/workers/fsdp_workers.py** (line 416)
   - here, we should return all metrics
442. **luffy/verl/verl/workers/fsdp_workers.py** (line 811)
   - we may need to extract it to dp_reward_model.py
443. **luffy/verl/verl/workers/megatron_workers.py** (line 106)
   - Currently, we only support reference model param offload
444. **luffy/verl/verl/workers/megatron_workers.py** (line 204)
   - add more optimizer args into config
445. **luffy/verl/verl/workers/megatron_workers.py** (line 338)
   - here, we should return all metrics
446. **luffy/verl/verl/workers/megatron_workers.py** (line 444)
   - support critic model offload
447. **luffy/verl/verl/workers/megatron_workers.py** (line 478)
   - support vpp here
448. **luffy/verl/verl/workers/megatron_workers.py** (line 507)
   - add more optimizer args into config
449. **luffy/verl/verl/workers/megatron_workers.py** (line 667)
   - add more optimizer args into config
450. **luffy/verl/verl/workers/megatron_workers.py** (line 720)
   - reward model use itself tokenizer instead of sft tokenizer
451. **luffy/verl/verl/workers/reward_model/megatron/reward_model.py** (line 145)
   - check why is bfloat16
452. **luffy/verl/verl/workers/reward_model/megatron/reward_model.py** (line 192)
   - actually, we just need to control the sampling order.
453. **luffy/verl/verl/workers/reward_model/megatron/reward_model.py** (line 233)
   - we may use the new schedule instead
454. **luffy/verl/verl/workers/rollout/hf_rollout.py** (line 98)
   - filter out the seq with no answers like ds-chat
455. **luffy/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py** (line 43)
   - implementation needed
456. **luffy/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 49)
   - check how to set seed for each model
457. **luffy/verl/verl/workers/sharding_manager/fsdp_ulysses.py** (line 56)
   - check how to set seed for each model
458. **luffy/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 82)
   - offload FSDP model weights
459. **luffy/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 113)
   - Current impl doesn't consider FSDP with torch micro-dp
460. **luffy/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 122)
   - Current impl doesn't consider FSDP with torch micro-dp
461. **luffy/verl/verl/workers/sharding_manager/fsdp_vllm.py** (line 130)
   - shall we build a micro_dp group for vllm when integrating with vLLM?
462. **luffy/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 76)
   - after binding to the memory buffer, we can load the checkpoint here
463. **luffy/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 253)
   - this may not be true for FSDP -> vLLM
464. **luffy/verl/verl/workers/sharding_manager/megatron_vllm.py** (line 323)
   - (zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.



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


