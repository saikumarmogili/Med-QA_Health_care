---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:10496
- loss:MultipleNegativesRankingLoss
base_model: pritamdeka/S-PubMedBert-MS-MARCO
widget:
- source_sentence: What causes Proctitis ?
  sentences:
  - Herpes-induced proctitis may be particularly severe in people who are also infected
    with the HIV virus.  - Non-STD infections. Infections that are not sexually transmitted
    also can cause proctitis. Salmonella and Shigella are examples of foodborne bacteria
    that can cause proctitis.
  - The most severe forms of the disorder tend to occur before birth and in early
    infancy. Hypophosphatasia weakens and softens the bones, causing skeletal abnormalities
    similar to another childhood bone disorder called rickets. Affected infants are
    born with short limbs, an abnormally shaped chest, and soft skull bones.
  - In come cases, symptoms are not dramatically apparent for years. Sleep studies
    are an essential part of the evaluation of people with possible narcolepsy. The
    combination of an overnight polysomnogram (PSG) followed by a multiple sleep latency
    test (MSLT) can provide strongly suggestive evidence of narcolepsy, while excluding
    other sleep disorders.
- source_sentence: What are the genetic changes related to pseudocholinesterase deficiency
    ?
  sentences:
  - Acanthamoeba is a microscopic, free-living ameba (single-celled living organism)
    commonly found in the environment that can cause rare, but severe, illness. Acanthamoeba
    causes three main types of illness involving the eye (Acanthamoeba keratitis),
    the brain and spinal cord (Granulomatous Encephalitis), and infections that can
    spread throughout the entire body (disseminated infection).
  - Treatment outcome is different for every person.
  - Other mutations prevent the production of the pseudocholinesterase enzyme. A lack
    of functional pseudocholinesterase enzyme impairs the body's ability to break
    down choline ester drugs efficiently, leading to abnormally prolonged drug effects.
    Pseudocholinesterase deficiency can also have nongenetic causes.
- source_sentence: How to prevent Prescription and Illicit Drug Abuse ?
  sentences:
  - Medicare drug plans are run by insurance companies and other private companies
    approved by Medicare. A person who joins Original Medicare and who wants prescription
    drug coverage will need to choose and sign up for a Medicare Prescription Drug
    plan (PDP). A person who joins one of the Medicare Advantage Plans will automatically
    receive prescription drug coverage through that plan if it's offered, usually
    for an extra cost.
  - Do not use other people's prescription medications, and do not share yours. Talk
    with your doctor before increasing or decreasing the medication dosage. - Do not
    stop taking a medicine on your own. Talk to your doctor if you are having side
    effects or other problems.  - Learn about the medicines possible interactions
    with alcohol and other prescription and over-the-counter medicines, and follow
    your doctors instructions to avoid these interactions.   - Answer honestly if
    a doctor or other health care professional asks you about other drug or alcohol
    use.
  - However, there is no safe and effective vaccine currently available for human
    use. Further research is needed to develop these potential vaccines as well as
    determine the efficacy of different treatment options including ribavirin and
    other antiviral drugs.
- source_sentence: What to do for Lactose Intolerance ?
  sentences:
  - This amount is the amount of lactose in 1 cup of milk.  - Many people can manage
    the symptoms of lactose intolerance by changing their diet. Some people may only
    need to limit the amount of lactose they eat or drink. Others may need to avoid
    lactose altogether.  - People may find it helpful to talk with a health care provider
    or a registered dietitian to determine if their diet provides adequate nutrients
    including calcium and vitamin D.
  - 'How is central core disease diagnosed? Because the symptoms of central core disease
    can be quite variable, a physical examination alone is often not enough to establish
    a diagnosis. A combination of the following examinations and testings can diagnosis
    this condition: a physical examination that confirms muscle weakness, a muscle
    biopsy that reveals a characteristic appearance of the muscle cells, and/or genetic
    testing that identifies a mutation in the RYR1.'
  - Klinefelter syndrome (KS) is a condition that occurs in men who have an extra
    X chromosome. The syndrome can affect different stages of physical, language,
    and social development. The most common symptom is infertility.
- source_sentence: Is phosphoribosylpyrophosphate synthetase superactivity inherited
    ?
  sentences:
  - This condition is inherited in an X-linked pattern. The gene associated with this
    condition is located on the X chromosome, which is one of the two sex chromosomes.
    In males (who have only one X chromosome), a mutation in the only copy of the
    gene in each cell causes the disorder.
  - AMP deaminase deficiency is one of the most common inherited muscle disorders
    in white populations, affecting 1 in 50 to 100 people. The prevalence is lower
    in African Americans, affecting an estimated 1 in 40,000 people, and the condition
    is even less common in the Japanese population.
  - This condition is inherited in an autosomal recessive pattern, which means both
    copies of the gene in each cell have mutations. The parents of an individual with
    an autosomal recessive condition each carry one copy of the mutated gene, but
    they typically do not show signs and symptoms of the condition.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on pritamdeka/S-PubMedBert-MS-MARCO
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dev
      type: dev
    metrics:
    - type: cosine_accuracy@1
      value: 0.660952380952381
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7923809523809524
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.832
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8689523809523809
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.660952380952381
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.4115555555555555
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.3080380952380953
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.196152380952381
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.23892183826436328
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.37751581346006674
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.4381989978674829
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.5130714875381451
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.5217865376857602
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.7341603930461064
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.43459909990508006
      name: Cosine Map@100
---

# SentenceTransformer based on pritamdeka/S-PubMedBert-MS-MARCO

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) <!-- at revision 96786c7024f95c5aac7f2b9a18086c7b97b23036 -->
- **Maximum Sequence Length:** 350 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 350, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Is phosphoribosylpyrophosphate synthetase superactivity inherited ?',
    'This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), a mutation in the only copy of the gene in each cell causes the disorder.',
    'AMP deaminase deficiency is one of the most common inherited muscle disorders in white populations, affecting 1 in 50 to 100 people. The prevalence is lower in African Americans, affecting an estimated 1 in 40,000 people, and the condition is even less common in the Japanese population.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6155, 0.3417],
#         [0.6155, 1.0000, 0.3496],
#         [0.3417, 0.3496, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dev`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.661      |
| cosine_accuracy@3   | 0.7924     |
| cosine_accuracy@5   | 0.832      |
| cosine_accuracy@10  | 0.869      |
| cosine_precision@1  | 0.661      |
| cosine_precision@3  | 0.4116     |
| cosine_precision@5  | 0.308      |
| cosine_precision@10 | 0.1962     |
| cosine_recall@1     | 0.2389     |
| cosine_recall@3     | 0.3775     |
| cosine_recall@5     | 0.4382     |
| cosine_recall@10    | 0.5131     |
| **cosine_ndcg@10**  | **0.5218** |
| cosine_mrr@10       | 0.7342     |
| cosine_map@100      | 0.4346     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 10,496 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 6 tokens</li><li>mean: 12.88 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 67.17 tokens</li><li>max: 350 tokens</li></ul> |
* Samples:
  | sentence_0                                                       | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:-----------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What causes X-linked adrenal hypoplasia congenita ?</code> | <code>The NR0B1 gene provides instructions to make a protein called DAX1. This protein plays an important role in the development and function of several hormone-producing tissues including the adrenal glands, two hormone-secreting glands in the brain (the hypothalamus and pituitary), and the gonads (ovaries in females and testes in males). The hormones produced by these glands control many important body functions.</code> |
  | <code>What is (are) Bell's Palsy ?</code>                        | <code>With or without treatment, most people begin to get better within 2 weeks and recover completely within 3 to 6 months. NIH: National Institute of Neurological Disorders and Stroke</code>                                                                                                                                                                                                                                           |
  | <code>What are the treatments for Sotos Syndrome ?</code>        | <code>There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.</code>                                                                                                                                                                                                                                                                                                                                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | dev_cosine_ndcg@10 |
|:-----:|:----:|:------------------:|
| 0.5   | 164  | 0.5148             |
| 1.0   | 328  | 0.5218             |


### Framework Versions
- Python: 3.12.11
- Sentence Transformers: 5.1.0
- Transformers: 4.55.2
- PyTorch: 2.8.0+cu126
- Accelerate: 1.10.0
- Datasets: 4.0.0
- Tokenizers: 0.21.4

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->