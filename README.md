# nn_text_compressor

This is a just a testing solution of idea.
Can a text file be compressed by NN ( Neural Network ).
Of course the learned model is still large but the idea is tempting to do some research in this field.


if some libs are missing .. remember just install them :)

 pip install torch, tqdm 


The sample file I created using this generator
    https://www.blindtextgenerator.com/lorem-ipsum
Thx for it.


#1# Training starts with 

  python .\nn_text_compressor.py --mode train --train_files sample_1.txt sample_2.txt --trained_model MODELLO.nn

  U will get MODELLO.nn file




#2# Compression with
 
  python .\nn_text_compressor.py --mode compress --trained_model MODELLO.nn --infile input_test_file.txt --outfile compressed_file.comp


#3# Decopression with 

 python .\nn_text_compressor.py --mode decompress --trained_model .\MODELLO.nn --infile compressed_file.comp --outfile decompressed.txt


If all works well it should be 
input_test_file.txt == decompressed.txt


