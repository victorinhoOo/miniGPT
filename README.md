# miniGPT

Une implémentation d'un modèle de langage de type GPT entraîné sur des données Wikipédia en français.

## 📋 Sommaire

* [Description](#description)
* [Architecture](#architecture)
* [Dataset](#dataset)
* [Entraînement](#entraînement)
* [Génération](#génération)
* [Installation](#installation)

## Description

Ce projet implémente un modèle de langage génératif basé sur l'architecture Transformer, similaire à GPT-2. Le modèle est entraîné sur un corpus de textes français issus de Wikipédia et peut générer du texte cohérent en français.

## Architecture

## Dataset

Le dataset est composé d'articles Wikipédia en français :

### Caractéristiques
- **Source** : Wikipedia FR (dump 2022)
- **Format** : Fichiers .npy contenant des tokens
- **Tokenizer** : tiktoken (compatible GPT-2)
- **Vocabulaire** : 50304 tokens

### Structure
- **Train** : Shards d'entraînement (train-*.npy)
- **Val** : Shards de validation (val-*.npy)
- **Taille des shards** : ~10M tokens par fichier
