```
  __  __    U  ___ u  _   _     _____  U _____ u      __  __      _       ____     ____     
U|' \/ '|u   \/"_ \/ | \ |"|   |_ " _| \| ___"|/    U|' \/ '|uU  /"\  u U|  _"\ u / __"| u  
\| |\/| |/   | | | |<|  \| |>    | |    |  _|"      \| |\/| |/ \/ _ \/  \| |_) |/<\___ \/   
 | |  | |.-,_| |_| |U| |\  |u   /| |\   | |___       | |  | |  / ___ \   |  __/   u___) |   
 |_|  |_| \_)-\___/  |_| \_|   u |_|U   |_____|      |_|  |_| /_/   \_\  |_|      |____/>>  
<<,-,,-.       \\    ||   \\,-._// \\_  <<   >>     <<,-,,-.   \\    >>  ||>>_     )(  (__) 
 (./  \.)     (__)   (_")  (_/(__) (__)(__) (__)     (./  \.) (__)  (__)(__)__)   (__)      
```

https://github.com/user-attachments/assets/3c1a2c6b-95ce-45ae-91c1-aee83f7ff574

MonteMaps is a [Google Maps](https://fr.wikipedia.org/wiki/Google_Maps)-like software that allows the user to localize themselves in the [Montefiore Institute](https://www.montefiore.uliege.be/cms/c_3482888/en/montefiore-institute) (ULiege) using a off-the-self camera.
The data created for this project (point cloud, mapping images, descriptors) can be obtained on [HuggingFace](https://huggingface.co/datasets/simon-gardier/monte-maps).

## Initial installation
1. **Download the submodules**
```bash
git submodule update --init --recursive
```

2. **Download env**
```bash
uv sync
```

3. **Hugging face**

If needed:
```bash
uv pip install huggingface_hub
hf auth login
```

4. **Data**

Download the project data (images, descriptors, camera poses,...) hosted on HuggingFace [monte-maps](https://huggingface.co/datasets/simon-gardier/monte-maps/tree/main).

5. **.env setup**

Copy `template.env` in a `.env` file, fill in the variables for the sections you are interested in (e.g if you want to test the realtime result, *Real Time localization* and *Global* sections need to be filled in).

**Note:** You need to download corresponding data from the [HuggingFace dataset]((https://huggingface.co/datasets/simon-gardier/monte-maps/tree/main)) and point env path variables to download location.

## Reconstruction

### COLMAP
#### Downloads
1. Download COLMAP GUI from: https://github.com/colmap/colmap/releases/tag/4.0.4
2. Download vocab tree "flickr100k_words256K" from: https://github.com/ZachMckennedyFWig/ColmapFaissVocabTrees/
3. Download `reconstructio-video-images-10fps.zip` (images from Montefiore Institute) from: [monte-maps](https://huggingface.co/datasets/simon-gardier/monte-maps/tree/main) to reconstruct the model yourself or download `model/` to use the precomputed model.

#### Reconstruction
Reconstruction can either be done manually or by running in sequence the "**reconstruction_...sh**" scripts from the `scripts/` folder  (these scripts require you to `source .env` before execution).

## Image Retrieval
Download the MegaLoc weights with `hf download gberton/MegaLoc model.safetensors --local-dir SOME_PATH` and set the related variable in the .env file 

## Real Time localization
From the project root:
- For simulation mode (requires a mp4 video), run `python src/monte-maps/main.py --mode simulation`
- For webcam mode, run `python src/monte-maps/main.py --mode webcam`

## Credits
- [Lei Yang](https://www.linkedin.com/in/lei-yang-05a37b26b/)
- [Camille Trinh](https://www.linkedin.com/in/camille-trinh-25b408349/)
- [Simon Gardier](https://github.com/simon-gardier/)

## Acknowledgement
Thanks to Pr. Anthony Cioppa for lending us their camera to capture the building.
