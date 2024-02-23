# Python scripts for data analysis in WAAM acoustic test
This folder has a bench of scripts, which are used to do research for fitting for acoustic data and height difference data

## Python version 
Python 3.10 is suggested for these scripts.

## Scripts introduction
* analysis_function.py - general method to analysis acoustic data and do ML with height data
* data_fft_transfer.py - A method to calculate FFT on exist acoustic data
* data_filter.py - Filter for certain range of frequency
* data_plot_audio_flipped.py - Flip audio data every 2 layers and plot, suitable for data input to model
* data_plot_height_flipped.py - Flip height difference every 2 layers and plot, suitable for data display for people
* data_pretreat_pipline.py - Norminal method for extracting data from test folder and save features into dictionary
* data_stft.py - Do STFT on exist acoustic data
* dictionary_vis.py - Plots and visualization of saved dictionary data
* fcnn_model_trian.py - FCNN model implementation
* mlp_model_train,py - MLP model implementation
* mlp_vis.py - Visulization of model performance
* plt_all_ml.py - TEST SCRIPTS, USELESS!