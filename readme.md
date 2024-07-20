# Pool Game Tracking System
This project contains a pool game tracking system developed by Alessandro Bozzon, Federico Adami and Simone Peraro.

## Requirements
The project requires:

- CMake 3.5 or above
- C++11 or above
- OpenCV 4.8.0 or above

## Project structure
The project folder has te followiing structure:

- CMakeLists.txt file to configure the project building
- /include folder, containing headers to project's classes
- /src folder, containing code source files
- /res/assets folder, containing images used to show a schematic drawing of the game
- the project requirements
- the report of the project

In addition, also a copy of the final output of the system is provided, after being run in live and evaluation modes (see below).

## Building instructions

1. Create a new building directory inside the project folder

```bash
mkdir build & cd build
```

2. Run CMake to configure the project
```bash
cmake ..
```

3. Compile the source files using make
```bash
make
```

## Running the system

The system can run in two modes (from build directory): 

- a live mode, providing path to a game clip video and to a folder to write directory
```bash [example]
./run_live path/to/dataset/gameclipfolder/gameclip.mp4 path/to/output/folder/
```
- an evaluation mode, to compute the required metrics providing a true game clip folder and an output folder

```bash
./run_evaluation path/to/gameclip/folder/ path/to/output/folder
```


Also, a simple executable is provided to show two masks files, remapped in more visible colors (can be a true and a predicted masks or any two other masks)

```bash
./open_mask path/to/mask/file.png path/to/antoher/mask/file.png
```

## Output provided examples
The project also provides outputs for each game clip already generated in `/results/` folder. It is possible to regenerate those runnig the following commands:
```bash
./run_live path/to/dataset/gameX_clipY/gameX_clipY.mp4 ../results/gameX_clipY/;
./run_evaluation path/to/dataset/gameX_clipY/ ../results/gameX_clipY/;
```
Tip: running `./run_evaluation` first will create the `gameX_clipY/` folder (if not existing), which can then be used as destination folder for `./run_live` command
