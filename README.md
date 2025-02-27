# miniGPT
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Model Size](https://img.shields.io/badge/model_size-125M-lightgrey)
![Dataset](https://img.shields.io/badge/dataset-FineWeb--Edu-blue)

This repository contains the code to train autoregressive language models like ChatGPT from scratch.

I used it to train a GPT-2 style model with two training phases:
1. Pre-training on FineWeb Edu dataset for general language understanding
2. Finetuning for conversational abilities on the openAssistant dataset (work in progress)

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
			<td align="center">1024</td>
		</tr>
		<tr>
			<td align="left">Number of layers</td>
			<td align="center">24</td>
		</tr>
		<tr>
			<td align="left">Number of heads</td>
			<td align="center">16</td>
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

The resulting model has 350M trainable parameters

<br/>

## Training

### Phase 1: Pre-training
- **Dataset**: FineWeb Edu
- **Purpose**: Learn general language understanding
- **Training time**: 5 days on A100 using Lambda Cloud https://cloud.lambdalabs.com/
- **Output**: Base model checkpoint

### Phase 2: Finetuning
WIP, will update later.

  ## Generation

  Here are some of the results I get when prompting the model with "Python is a programming language". <br> Right now the model can't respond to questions, but I'm hoping it will be soon once finetuning is done ! 

  ```console
Sample 1: Python is a programming language designed by Guido van Rossum. 
It is a general purpose programming language with a wide range of applications. 
In Python, variables and functions are organized in packages and functions are 
named in the format that the programmer chooses.
```
```console
Sample 2: Python is a programming language that many consider to be the most popular 
one in the world, and the one that is most often used in web development today. 
It is known for being fast and being able to handle large amounts of data.
```
```console
Sample 3: Python is a programming language that is used to create applications, 
web servers and other programming platforms. Python is one of the most 
popular programming languages, especially among programmers who use it to 
create web sites and applications.
```


