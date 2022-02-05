It is recommended to put training data in this folder, and CRLT only supports csv file format. In addition, CRLT supports a variety of training data formats:
+ single-column, CRLT generates `query` and `key` according to the configuration. 
+ double-column, CRLT uses paired data as `query` and `key`.
+ three-column, If hard negative data is available, the `negative` column must be included in the CSV file.