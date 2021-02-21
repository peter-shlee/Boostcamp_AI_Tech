# Natural Language Processing
## Assignment 2: Training NMT model with fairseq

### 1. Introduction

- 본 과제의 목적은 대표적인 pytorch library 중 하나인 fairseq을 이용해 번역 모델을 학습하는 방법을 배우는 것입니다.
- 일반적으로 우리는 해당 task와 관련되어 널리 알려진 library를 이용해 모델을 구현하고 tuning하게 됩니다. 자연어 처리 관련 여러 라이브러리가 있지만 번역 task에서 가장 자주 활용되고 있는 fairseq을 소개해드리고자 합니다. fairseq은 pytorch를 개발하고 있는 facebook에서 작업 중인 오픈소스 프로젝트 입니다. library의 이름처럼 sequence를 다루는 데에 필요한 여러 모델과 전처리, 평가 관련 코드를 포함해 인기가 많은 library 중 하나입니다.
- 이번 시간에는 해당 library의 [docs](https://fairseq.readthedocs.io/en/latest/)를 직접 읽어보면서 목표 perplexity/BLEU를 달성하는 것이 목표입니다. 프로젝트에 대한 대략적인 설명과 관련 docs 링크를 함께 제공해드리겠습니다. 주어진 데이터에 대해 **BLEU score 25 이상**을 달성해보세요 !
- ***먼저 colab 상단 탭 runtime -> change runtime type에서 hardware accelerator를 GPU로 변경해주세요***


```python
# 필요 패키지 설치 (참고: 느낌표를 앞에 붙이면 해당 코드가 terminal에서 실행됩니다.)
!pip install fastBPE sacremoses subword_nmt hydra-core omegaconf fairseq
```

    Requirement already satisfied: fastBPE in /usr/local/lib/python3.6/dist-packages (0.1.0)
    Requirement already satisfied: sacremoses in /usr/local/lib/python3.6/dist-packages (0.0.43)
    Requirement already satisfied: subword_nmt in /usr/local/lib/python3.6/dist-packages (0.3.7)
    Requirement already satisfied: hydra-core in /usr/local/lib/python3.6/dist-packages (1.0.6)
    Requirement already satisfied: omegaconf in /usr/local/lib/python3.6/dist-packages (2.0.6)
    Requirement already satisfied: fairseq in /usr/local/lib/python3.6/dist-packages (0.10.2)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses) (1.15.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from sacremoses) (4.41.1)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses) (1.0.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses) (7.1.2)
    Requirement already satisfied: regex in /usr/local/lib/python3.6/dist-packages (from sacremoses) (2019.12.20)
    Requirement already satisfied: antlr4-python3-runtime==4.8 in /usr/local/lib/python3.6/dist-packages (from hydra-core) (4.8)
    Requirement already satisfied: importlib-resources; python_version < "3.9" in /usr/local/lib/python3.6/dist-packages (from hydra-core) (5.1.0)
    Requirement already satisfied: PyYAML>=5.1.* in /usr/local/lib/python3.6/dist-packages (from omegaconf) (5.4.1)
    Requirement already satisfied: dataclasses; python_version == "3.6" in /usr/local/lib/python3.6/dist-packages (from omegaconf) (0.8)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.6/dist-packages (from omegaconf) (3.7.4.3)
    Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from fairseq) (1.7.0+cu101)
    Requirement already satisfied: cython in /usr/local/lib/python3.6/dist-packages (from fairseq) (0.29.21)
    Requirement already satisfied: sacrebleu>=1.4.12 in /usr/local/lib/python3.6/dist-packages (from fairseq) (1.5.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from fairseq) (1.19.5)
    Requirement already satisfied: cffi in /usr/local/lib/python3.6/dist-packages (from fairseq) (1.14.4)
    Requirement already satisfied: zipp>=0.4; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from importlib-resources; python_version < "3.9"->hydra-core) (3.4.0)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch->fairseq) (0.16.0)
    Requirement already satisfied: portalocker in /usr/local/lib/python3.6/dist-packages (from sacrebleu>=1.4.12->fairseq) (2.2.1)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.6/dist-packages (from cffi->fairseq) (2.20)



```python
# clone fairseq git
!git clone https://github.com/pytorch/fairseq.git
# iwslt14 데이터 준비 (https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh)
!bash fairseq/examples/translation/prepare-iwslt14.sh
```

    fatal: destination path 'fairseq' already exists and is not an empty directory.
    Cloning Moses github repository (for tokenization scripts)...
    fatal: destination path 'mosesdecoder' already exists and is not an empty directory.
    Cloning Subword NMT repository (for BPE pre-processing)...
    fatal: destination path 'subword-nmt' already exists and is not an empty directory.
    Downloading data from http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz...
    --2021-02-17 10:03:17--  http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz
    Resolving dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)... 172.67.9.4, 104.22.75.142, 104.22.74.142, ...
    Connecting to dl.fbaipublicfiles.com (dl.fbaipublicfiles.com)|172.67.9.4|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19982877 (19M) [application/x-tar]
    Saving to: ‘de-en.tgz.2’
    
    de-en.tgz.2         100%[===================>]  19.06M  16.7MB/s    in 1.1s    
    
    2021-02-17 10:03:19 (16.7 MB/s) - ‘de-en.tgz.2’ saved [19982877/19982877]
    
    Data successfully downloaded.
    de-en/
    de-en/IWSLT14.TED.dev2010.de-en.de.xml
    de-en/IWSLT14.TED.dev2010.de-en.en.xml
    de-en/IWSLT14.TED.tst2010.de-en.de.xml
    de-en/IWSLT14.TED.tst2010.de-en.en.xml
    de-en/IWSLT14.TED.tst2011.de-en.de.xml
    de-en/IWSLT14.TED.tst2011.de-en.en.xml
    de-en/IWSLT14.TED.tst2012.de-en.de.xml
    de-en/IWSLT14.TED.tst2012.de-en.en.xml
    de-en/IWSLT14.TEDX.dev2012.de-en.de.xml
    de-en/IWSLT14.TEDX.dev2012.de-en.en.xml
    de-en/README
    de-en/train.en
    de-en/train.tags.de-en.de
    de-en/train.tags.de-en.en
    pre-processing train data...
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    clean-corpus.perl: processing iwslt14.tokenized.de-en/tmp/train.tags.de-en.tok.de & .en to iwslt14.tokenized.de-en/tmp/train.tags.de-en.clean, cutoff 1-175, ratio 1.5
    ..........(100000).......
    Input sentences: 174443  Output sentences:  167522
    pre-processing valid/test data...
    orig/de-en/IWSLT14.TED.dev2010.de-en.de.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.dev2010.de-en.de
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2010.de-en.de.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2010.de-en.de
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2011.de-en.de.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2011.de-en.de
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2012.de-en.de.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2012.de-en.de
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    orig/de-en/IWSLT14.TEDX.dev2012.de-en.de.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TEDX.dev2012.de-en.de
    Tokenizer Version 1.1
    Language: de
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.dev2010.de-en.en.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.dev2010.de-en.en
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2010.de-en.en.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2010.de-en.en
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2011.de-en.en.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2011.de-en.en
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    orig/de-en/IWSLT14.TED.tst2012.de-en.en.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TED.tst2012.de-en.en
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    orig/de-en/IWSLT14.TEDX.dev2012.de-en.en.xml iwslt14.tokenized.de-en/tmp/IWSLT14.TEDX.dev2012.de-en.en
    Tokenizer Version 1.1
    Language: en
    Number of threads: 8
    
    creating train, valid, test...
    learn_bpe.py on iwslt14.tokenized.de-en/tmp/train.en-de...
    apply_bpe.py to train.de...
    apply_bpe.py to valid.de...
    apply_bpe.py to test.de...
    apply_bpe.py to train.en...
    apply_bpe.py to valid.en...
    apply_bpe.py to test.en...


## 2. Library reference
1. [tasks](https://fairseq.readthedocs.io/en/latest/tasks.html)
 - translation task와 language modeling task가 있고 나머지 sequence를 다루는 task는 register_task() function decorator를 이용해 등록할 수 있습니다.
2. [models](https://fairseq.readthedocs.io/en/latest/models.html)
 - 모델은 CNN, LSTM, Transformer 기반 모델들이 분류가 되어 있습니다. transformer 모델쪽 코드가 꼼꼼히 잘되어 있습니다. 새로운 모델을 등록하기 위해서는 register_model() function decorator를 이용할 수 있습니다.
3. [criterions](https://fairseq.readthedocs.io/en/latest/criterions.html)
 - 모델 학습을 위한 다양한 loss들이 구현되어 있습니다.
4. [optimizers](https://fairseq.readthedocs.io/en/latest/optim.html)
 - 모델 학습을 위한 다양한 optimizer들이 구현되어 있습니다.
5. l[earning rate schedulers](https://fairseq.readthedocs.io/en/latest/lr_scheduler.html)
 - 모델의 더 나은 학습을 위한 다양한 learning rate scheduler들이 구현되어 있습니다.
6. [data loading and utilities](https://fairseq.readthedocs.io/en/latest/data.html)
 - 전처리 및 데이터 관련 다양한 class들이 구현되어 있습니다.
7. [modules](https://fairseq.readthedocs.io/en/latest/modules.html)
 - 앞의 6군데에 속하지 못한(?) 다양한 모듈들이 구현되어 있습니다.

## 3. [Command-line Tools](https://fairseq.readthedocs.io/en/latest/command_line_tools.html)
fairseq은 학습과 평가를 쉽게할 수 있는 command-line tool을 제공하고 있습니다
각각의 커맨드라인에 대한 설명은 위 링크에 자세히 나와있으니 참고해주시기 바랍니다.
1. fairseq-preprocess
 - 데이터 학습을 위한 vocab을 만들고 data를 구성합니다.
2. fairseq-train
 - 여러 gpu 또는 단일 gpu에서 모델을 학습시킵니다.
3. fairseq-generate
 - 학습된 모델을 이용해 전처리된 데이터를 번역합니다.
4. fairseq-interactive
 - 학습된 모델을 이용해 raw 데이터를 번역합니다.
5. fairseq-score
 - 학습된 모델이 생성한 문장과 정답 문장을 비교해 bleu score를 산출합니다.
6. fairseq-eval-lm
 - language model을 평가할 수 있는 command입니다.



```python
# 예시 코드를 이용해 직접 전처리부터 평가까지 진행해보겠습니다.
# source-lang: source language
# target-lang: target language
# trainpref: train file prefix
# validpref: valid file prefix
# testpref: test file prefix
# destdir: destination dir
!fairseq-preprocess --source-lang de --target-lang en --trainpref ./iwslt14.tokenized.de-en/train --validpref ./iwslt14.tokenized.de-en/valid --testpref ./iwslt14.tokenized.de-en/test --destdir ./iwslt14.tokenized.de-en/
```

    2021-02-17 10:04:34 | INFO | fairseq_cli.preprocess | Namespace(align_suffix=None, alignfile=None, all_gather_list_size=16384, bf16=False, bpe=None, checkpoint_shard_count=1, checkpoint_suffix='', cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='./iwslt14.tokenized.de-en/', empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_format=None, log_interval=100, lr_scheduler='fixed', memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, only_source=False, optimizer=None, padding_factor=8, profile=False, quantization_config_path=None, scoring='bleu', seed=1, source_lang='de', srcdict=None, target_lang='en', task='translation', tensorboard_logdir=None, testpref='./iwslt14.tokenized.de-en/test', tgtdict=None, threshold_loss_scale=None, thresholdsrc=0, thresholdtgt=0, tokenizer=None, tpu=False, trainpref='./iwslt14.tokenized.de-en/train', user_dir=None, validpref='./iwslt14.tokenized.de-en/valid', workers=1)
    2021-02-17 10:04:52 | INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
    2021-02-17 10:05:20 | INFO | fairseq_cli.preprocess | [de] ./iwslt14.tokenized.de-en/train.de: 160239 sents, 4035591 tokens, 0.0% replaced by <unk>
    2021-02-17 10:05:20 | INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
    2021-02-17 10:05:21 | INFO | fairseq_cli.preprocess | [de] ./iwslt14.tokenized.de-en/valid.de: 7283 sents, 182592 tokens, 0.0192% replaced by <unk>
    2021-02-17 10:05:21 | INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
    2021-02-17 10:05:22 | INFO | fairseq_cli.preprocess | [de] ./iwslt14.tokenized.de-en/test.de: 6750 sents, 161838 tokens, 0.0636% replaced by <unk>
    2021-02-17 10:05:22 | INFO | fairseq_cli.preprocess | [en] Dictionary: 6632 types
    2021-02-17 10:05:48 | INFO | fairseq_cli.preprocess | [en] ./iwslt14.tokenized.de-en/train.en: 160239 sents, 3949114 tokens, 0.0% replaced by <unk>
    2021-02-17 10:05:48 | INFO | fairseq_cli.preprocess | [en] Dictionary: 6632 types
    2021-02-17 10:05:49 | INFO | fairseq_cli.preprocess | [en] ./iwslt14.tokenized.de-en/valid.en: 7283 sents, 178622 tokens, 0.00448% replaced by <unk>
    2021-02-17 10:05:49 | INFO | fairseq_cli.preprocess | [en] Dictionary: 6632 types
    2021-02-17 10:05:51 | INFO | fairseq_cli.preprocess | [en] ./iwslt14.tokenized.de-en/test.en: 6750 sents, 156928 tokens, 0.00892% replaced by <unk>
    2021-02-17 10:05:51 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to ./iwslt14.tokenized.de-en/



```python
# 모델 학습
# (참고: 모델을 동시에 여러개 학습 시키고 싶으신 분들은 노트북 파일을 드라이브에 여러개 복사해서 각 파일마다 실행하면 여러 모델을 동시에 학습시킬 수 있습니다.)
# --arch: architecture
# --optimizer: optimizer {adadelta, adam, adafactor, adagrad, lamb, composite, nag, adamax, sgd}
# --clip-norm: clip threshold of gradients
# --lr: learning rate
# --lr-scheduler: learning rate scheduler {pass_through, cosine, reduce_lr_on_plateau, fixed, triangular, polynomial_decay, tri_stage, manual, inverse_sqrt}
# --criterion loss function {sentence_prediction, ctc, adaptive_loss, label_smoothed_cross_entropy, composite_loss, nat_loss, masked_lm, sentence_ranking, legacy_masked_lm_loss, cross_entropy, model, wav2vec, label_smoothed_cross_entropy_with_alignment, vocab_parallel_cross_entropy}
# --max-tokens: maximum number of tokens in a batch
# --max-epoch: maximum number of training epoch

!fairseq-train ./iwslt14.tokenized.de-en/ --arch transformer_iwslt_de_en --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --criterion label_smoothed_cross_entropy --max-tokens 4096 --max-epoch 3
```

    2021-02-17 05:30:20 | INFO | fairseq_cli.train | Namespace(activation_dropout=0.0, activation_fn='relu', adam_betas='(0.9, 0.999)', adam_eps=1e-08, 
    P-72	-0.9421 -0.5767 -0.0000 -2.7359 -0.8708 -0.0689 -0.2010 -1.0252 -0.8878 -1.8714 -0.3181 -3.7493 -2.0196 -0.6691 -0.2881 -1.2504 -1.6346 -0.0588 -0.0842 -0.5849 -0.0877 -0.3539 -0.0315 -5.1318 -1.6830 -0.3196 -0.5049 -2.4935 -0.7562 -0.1322 -0.3907 -0.0800 -0.1027 -0.8745 -0.1027 -0.1121 -1.2163 -1.1712 -0.0249 -0.
    
    (생략)

    2021-02-17 10:49:05 | INFO | fairseq_cli.generate | Translated 6750 sentences (136686 tokens) in 36.8s (183.53 sentences/s, 3716.49 tokens/s)
    Generate test with beam=5: BLEU4 = 26.48, 65.9/37.5/23.2/14.5 (BP=0.877, ratio=0.884, syslen=115893, reflen=131161)

