# Multilayer_GOMLP

This project focusing on extending the GOMLP to multilayer cases. The GOMLP (published in DAC 2022) solves 2D power plane generation problem with combination of MLP and evolutionary method. The proposed method for multilayer GOMLP is as follows:

<p align="center">
<img src="Fig/MultiGOMLP.png" alt="drawing" width="800">
</p>


Project strctures: 
```
gomlp
├── MLGO_2_with_plots.py
├── tmp
├── color_map.pkl
├── pin_file (e.g. pins_BeagleBone_RevC_human_singleLayer.csv) 
surrogate
├── surrogate_model_new.py
├── tmp
├── output.csv (e.g. output_50nets.csv)
├── color_map.pkl
├── pin_file (e.g. pins_BeagleBone_RevC_human_singleLayer.csv) 
utils
├── generate_testcases.py
```

Code running instructions: 
#### 1. Running (2D) GOMLP: 
File needed:
  - pin_file (e.g. pins_BeagleBone_RevC_human.csv _[pin config for all nets and layers])
  - colormap.pkl (optional, will automatically generate if not exists)
Under directory gomlp, specify the problems parameters (net name, layer name, etc. in MLGO_2_with_plots.py) and run the following 
```
python MLGO_2_with_plots.py 
```

For more info, refer to the original repo: https://github.com/vinayp173/GOMLP-2

#### 2. Running surrogate modoels: 



#### Reference: 
1. GOMLP DAC paper: "AUTOMATIC POWER PLANE GENERATION WITH GENETIC OPTIMIZATION AND MULTILAYER PERCEPTRON"
