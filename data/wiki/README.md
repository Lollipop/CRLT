We provide the wikipedia data collected by SimCSE and its variants:

+ `wiki.csv` file contains part of the `wiki1m_for_simcse` file where the sentence length is greater than a threshold (0.7M).
+ `wiki_paraphrased` is paired data, which paraphrased by `tuner007/pegasus_paraphrase` model using `wiki.csv` data.
+ `wiki_translated_99.0_36.54.csv` is also paired data, which uses german as intermediate language translated by `facebook/wmt19-en-de` and `facebook/wmt19-de-en` model.

|Filename | Data Path | Google Drive |
|:--------|:----------|:-----------:|
| wiki1m_for_simcse.csv | data/wiki/ | [Download](https://drive.google.com/file/d/1Wqtlczfs_6uUeVzrfO7vHQmGcJcytJ7z/view?usp=sharing) |
| wiki.csv | data/wiki/ | [Download](https://drive.google.com/file/d/1y_nS6lj32Asxb-aWKVuYWFQNC08eQJV5/view?usp=sharing) |
| wiki_paraphrased_99.0_70.31.csv| data/wiki/ | [Download](https://drive.google.com/file/d/12LO2dZHCM3XT2ZwEhJbAEpMe8aynfWw3/view?usp=sharing) |
| wiki_translated_99.0_36.54.csv | data/wiki/ | [Download](https://drive.google.com/file/d/1iNilTJ3PDvm2xjUclqkQOh34RD9jbJa0/view?usp=sharing) |