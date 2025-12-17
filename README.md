# nn_text_compressor

This is a just a idea testing solution of idea.
Can a text file be compressed by NN ( Neural Network )

if some libs are missing .. remember just install them :)

 pip install torch, tqdm 




The sample file I created using this generator
    https://www.blindtextgenerator.com/lorem-ipsum
Thx for it.

Training starts with 
python .\nn_text_compressor.py --mode train --train_files sample_1.txt sample_2.txt --trained_model MODELLO.nn

U will get MODELLO.nn file

Compression with
python .\nn_text_compressor.py --mode compress --trained_model MODELLO.nn --infile input_test_file.txt --outfile compressed_file.comp


Decopression with 

python .\nn_text_compressor.py --mode decompress --trained_model .\MODELLO.nn --infile compressed_file.comp --outfile decompressed.txt


If all works well it should be 
input_test_file.txt == decompressed.txt


