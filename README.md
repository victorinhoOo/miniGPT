# miniGPT
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Model Size](https://img.shields.io/badge/model_size-125M-lightgrey)
![Dataset](https://img.shields.io/badge/dataset-FineWeb--Edu-blue)

This repository contains the code to train and finetune autoregressive language models like ChatGPT from scratch.

I used it to train a GPT-2 style model with two training phases:
1. Pre-training on FineWeb Edu dataset
2. Finetuning for conversational abilities on Dolly Dataset

## 📋 Table of Contents

* [Description](#description)
* [Architecture](#architecture)
* [Dataset](#dataset)
* [Training](#training)
* [Generation](#generation)
* [Installation](#installation)

## Description

Using this repo i implemented a GPT-2 style language model with two stages:
- First trained on FineWeb Edu for general language understanding
- Then finetuned on instruction data to create a helpful assistant

But i can be easily adapted to your own datasets and needs. 


## 🏗️ Architecture

The model is based on the transformer architecture (only the decoder part) from the paper [**Attention is All You Need**](https://doi.org/10.48550/arXiv.1706.03762) by **Google Brain** (2017), with a few improvements:


* I moved the normalization layers before the transformer blocks (instead of after) like in the paper [**On Layer Normalization in the Transformer Architecture**](https://doi.org/10.48550/arXiv.2002.04745) by **Microsoft Research** (2020)

* I replaced the ReLU activation by the GeLU activation from the paper [**GLU Variants Improve Transformer**](https://doi.org/10.48550/arXiv.2002.05202) by **Google** (2020)




<br/>

Here are the main parameters of the architecture i used for my own implementation :

<table>
	<thead>
		<tr>
			<th align="center">Parameter</th>
			<th align="center">Value</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td align="left">Embedding dimension</td>
			<td align="center">768</td>
		</tr>
		<tr>
			<td align="left">Number of layers</td>
			<td align="center">12</td>
		</tr>
		<tr>
			<td align="left">Number of heads</td>
			<td align="center">12</td>
		</tr>
		<tr>
			<td align="left">Context length</td>
			<td align="center">1024</td>
		</tr>
		<tr>
			<td align="left">Vocab size</td>
			<td align="center">50257</td>
		</tr>
	</tbody>
</table>

The resulting model has 124M trainable parameters

<br/>

## Training Process

### Phase 1: Pre-training
- **Dataset**: FineWeb Edu
- **Purpose**: Learn general language understanding
- **Training time**: 4 days on A100 using Lambda Cloud https://cloud.lambdalabs.com/
- **Output**: Base model checkpoint

### Phase 2: Finetuning
- **Dataset**: databricks/dolly-15k 
- **Purpose**: Convert to conversational assistant
- **Training time**: ~30 minutes on A100
- **Output**: Final assistant model

## Datasets

### FineWeb Edu (Pre-training)
- High-quality multilingual text corpus
- Processed into .npy tokenized files
- Used for general language capabilities

### Instruction Dataset (Finetuning)
- ~50K question-answer pairs
- Focused on helpful dialogue
- Used to teach conversational abilities

